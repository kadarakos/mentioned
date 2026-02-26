import torch
import torchmetrics


from transformers import AutoTokenizer, AutoModel
from huggingface_hub import PyTorchModelHubMixin
from lightning import LightningModule


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "distilroberta-base",
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.dim = self.encoder.config.hidden_size
        self.stats = {}

    def forward(self, input_ids, attention_mask, word_ids):
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            word_ids: (batch, seq_len) -> Pre-computed word indices,
                      use -1 for special tokens/padding.
        """
        # 1. Get Transformer Output
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        subword_embeddings = outputs.last_hidden_state  # (B, S, D)
        num_words = word_ids.max() + 1
        word_mask = word_ids.unsqueeze(-1) == torch.arange(
            num_words, device=word_ids.device
        )
        word_mask = word_mask.float()  # (B, S, W)
        # Sum embeddings for each word: (B, W, S) @ (B, S, D) -> (B, W, D)
        word_sums = torch.bmm(word_mask.transpose(1, 2), subword_embeddings)
        # Count subwords per word to get the denominator
        # (B, W, S) @ (B, S, 1) -> (B, W, 1)
        subword_counts = word_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        # (B, W, D)
        word_embeddings = word_sums / subword_counts
        return word_embeddings


class SwiGLU(torch.nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        # Common expansion factor
        if hidden_dim is None:
            hidden_dim = 2 * dim
        self.w1 = torch.nn.Linear(dim, hidden_dim)
        self.w3 = torch.nn.Linear(dim, hidden_dim)
        self.w2 = torch.nn.Linear(hidden_dim, dim)
        self.silu = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.silu(self.w1(x))
        x = gate * self.w3(x)
        x = self.w2(x)
        return x


class Detector(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # A 2-layer MLP is standard for span detection to capture interactions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),  # Output a single logit per token/pair
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, input_dim) for start detection
               (B, N, N, input_dim) for end detection
        Returns:
            logits: (B, N) or (B, N, N)
        """
        return self.net(x)


class MentionDetectorCore(torch.nn.Module):
    def __init__(
        self,
        start_detector: Detector,
        end_detector: Detector,
    ):
        super().__init__()
        self.start_detector = start_detector
        self.end_detector = end_detector

    def forward(self, emb: torch.Tensor):
        """
        Args:
            emb: (Batch, Seq_Len, Hidden_Dim)
        Returns:
            start_logits: (Batch, Seq_Len)
            end_logits:   (Batch, Seq_Len, Seq_Len)
        """
        B, N, H = emb.shape
        start_logits = self.start_detector(emb).squeeze(-1)
        start_rep = emb.unsqueeze(2).expand(-1, -1, N, -1)
        end_rep = emb.unsqueeze(1).expand(-1, N, -1, -1)
        pair_emb = torch.cat([start_rep, end_rep], dim=-1)
        end_logits = self.end_detector(pair_emb).squeeze(-1)

        return start_logits, end_logits


class LitMentionDetector(LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        tokenizer, #: transformers.PreTrainedTokenizer,
        encoder: torch.nn.Module,
        mention_detector: torch.nn.Module,
        lr: float = 2e-5,
        threshold: float = 0.5
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'start_detector', 'end_detector'])
        self.tokenizer = tokenizer
        self.encoder = encoder
        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mention_detector = mention_detector

        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Two separate metrics for the two tasks
        self.val_f1_start = torchmetrics.classification.BinaryF1Score()
        self.val_f1_end = torchmetrics.classification.BinaryF1Score()
        self.val_f1_mention = torchmetrics.classification.BinaryF1Score()

    def encode(self, docs: list[list[str]]):
        """
        Handles the non-vectorized tokenization and calls the vectorized encoder.
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(
            docs,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.encoder.max_length,
            padding=True,
            return_attention_mask=True,
            return_offsets_mapping=True  # needed for word_ids
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        batch_word_ids = []
        for i in range(len(docs)):
            w_ids = [w if w is not None else -1 for w in inputs.word_ids(batch_index=i)]
            batch_word_ids.append(torch.tensor(w_ids))

        word_ids_tensor = torch.stack(batch_word_ids).to(device)
        word_embeddings = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                word_ids=word_ids_tensor
            )
        return word_embeddings

    def forward(self, emb: torch.Tensor):
        start_logits, end_logits = self.mention_detector(emb)
        return start_logits, end_logits

    def _compute_start_loss(self, start_logits, batch):
        targets = batch["starts"].float()
        mask = batch["token_mask"].bool()
        return self.loss_fn(start_logits, targets)[mask].mean()

    def _compute_end_loss(self, end_logits, batch):
        targets = batch["spans"].float()
        mask = batch["span_loss_mask"].bool()
        raw_loss = self.loss_fn(end_logits, targets)
        relevant_loss = raw_loss[mask]

        if relevant_loss.numel() == 0:
            return end_logits.sum() * 0
        return relevant_loss.mean()

    def training_step(self, batch, batch_idx):
        emb = self.encode(batch["sentences"])
        start_logits, end_logits = self.forward(emb)
        loss_start = self._compute_start_loss(start_logits, batch)
        loss_end = self._compute_end_loss(end_logits, batch)
        total_loss = loss_start + loss_end
        self.log_dict({
            "train_loss": total_loss,
            "train_start_loss": loss_start,
            "train_end_loss": loss_end
        }, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        emb = self.encode(batch["sentences"])
        start_logits, end_logits = self.forward(emb)
        token_mask = batch["token_mask"].bool()
        
        # --- METRIC 1 & 2: Diagnostic metrics remain the same ---
        start_preds = (torch.sigmoid(start_logits[token_mask]) > self.hparams.threshold).int()
        start_targets = batch["starts"][token_mask].int()
        if start_targets.numel() > 0:
            self.val_f1_start.update(start_preds, start_targets)

        span_loss_mask = batch["span_loss_mask"].bool()
        end_preds_diag = (torch.sigmoid(end_logits[span_loss_mask]) > self.hparams.threshold).int()
        end_targets_diag = batch["spans"][span_loss_mask].int()
        if end_targets_diag.numel() > 0:
            self.val_f1_end.update(end_preds_diag, end_targets_diag)

        # --- METRIC 3: Full Mention Detection (Logical AND Fix) ---
        # 1. Get binary masks for starts and spans independently
        is_start = torch.sigmoid(start_logits) > self.hparams.threshold # (B, N)
        is_span = torch.sigmoid(end_logits) > self.hparams.threshold    # (B, N, N)

        # 2. Logical AND: A mention exists IF it's a valid start AND a valid span
        # (B, N, 1) & (B, N, N) -> (B, N, N)
        combined_preds_mask = is_start.unsqueeze(2) & is_span

        # 3. Apply structural masks (padding and upper triangle)
        valid_pair_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        upper_tri = torch.triu(torch.ones_like(end_logits), diagonal=0).bool()
        mention_eval_mask = valid_pair_mask & upper_tri

        # 4. Flatten for F1 calculation
        mention_preds = combined_preds_mask[mention_eval_mask].int()
        mention_targets = batch["spans"][mention_eval_mask].int()

        if mention_targets.numel() > 0:
            self.val_f1_mention.update(mention_preds, mention_targets)

        # --- 4. Logging ---
        start_loss = self._compute_start_loss(start_logits, batch)
        end_loss = self._compute_end_loss(end_logits, batch)
        self.log_dict({
            "val_loss": start_loss + end_loss,
            "val_f1_start": self.val_f1_start,
            "val_f1_end": self.val_f1_end,
            "val_f1_mention": self.val_f1_mention
        }, prog_bar=True, batch_size=len(batch["sentences"]), on_epoch=True)

    @torch.no_grad()
    def predict_mentions(self, sentences: list[list[str]], batch_size: int = 2) -> list[list[tuple[int, int]]]:
        self.eval()
        all_results = []

        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i : i + batch_size]
            emb = self.encode(batch_sentences) 
            start_logits, end_logits = self.forward(emb)

            # 1. Individual Boolean Decisions
            is_start = torch.sigmoid(start_logits) > self.hparams.threshold # (B, N)
            is_span = torch.sigmoid(end_logits) > self.hparams.threshold    # (B, N, N)

            # 2. Structural Constraint (j >= i)
            N = end_logits.size(1)
            upper_tri = torch.triu(torch.ones((N, N), device=self.device), diagonal=0).bool()

            # 3. Logical AND Gate
            # Mentions must: be a start AND be a span AND be upper triangle
            pred_mask = is_start.unsqueeze(2) & is_span & upper_tri

            # 4. Extract Indices
            indices = pred_mask.nonzero() # [batch_idx, start_idx, end_idx]

            batch_results = [[] for _ in range(len(batch_sentences))]
            for b_idx, s_idx, e_idx in indices:
                batch_results[b_idx.item()].append((s_idx.item(), e_idx.item()))

            all_results.extend(batch_results)

        return all_results


    def test_step(self, batch, batch_idx):
        # Reuse all the logic from validation_step
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

def make_model_v1(model_name="distilroberta-base"):
    dim = 768
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = SentenceEncoder(model_name).train()
    encoder.train()
    start_detector = Detector(dim, dim)
    end_detector = Detector(dim * 2, dim)
    mention_detector = MentionDetectorCore(start_detector, end_detector)
    return LitMentionDetector(tokenizer, encoder, mention_detector)
