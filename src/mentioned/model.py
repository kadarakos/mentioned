import torch
import torchmetrics

from transformers import AutoTokenizer, AutoModel
from huggingface_hub import PyTorchModelHubMixin
from lightning import LightningModule

from mentioned.data import DataBlob

class ModelRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name):
        return cls._registry[name]


class SentenceEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "distilroberta-base",
        max_length: int = 512,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,  # We need fast of the word_ids.
        )
        self.encoder = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.dim = self.encoder.config.hidden_size
        self.stats = {}

    def forward(self, input_ids, attention_mask, word_ids):
        """
        Args:
            input_ids: B x N
            attention_mask: B x N
            word_ids: B x N
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        subword_embeddings = outputs.last_hidden_state  # B x N x D
        num_words = word_ids.max() + 1
        word_mask = word_ids.unsqueeze(-1) == torch.arange(
            num_words, device=word_ids.device
        )
        word_mask = word_mask.to(subword_embeddings.dtype)
        # Sum embeddings for each word: (B, W, S) @ (B, S, D) -> (B, W, D)
        word_sums = torch.bmm(word_mask.transpose(1, 2), subword_embeddings)
        # Count subwords per word to get the denominator
        # (B, W, S) @ (B, S, 1) -> (B, W, 1)
        subword_counts = word_mask.sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        # (B, W, D)
        word_embeddings = word_sums / subword_counts
        return word_embeddings


class Detector(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 1,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
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
        # FIXME materialize all pairs is expensive.
        start_rep = emb.unsqueeze(2).expand(-1, -1, N, -1)
        end_rep = emb.unsqueeze(1).expand(-1, N, -1, -1)
        pair_emb = torch.cat([start_rep, end_rep], dim=-1)
        end_logits = self.end_detector(pair_emb).squeeze(-1)

        return start_logits, end_logits


class MentionLabeler(torch.nn.Module):
    def __init__(self, classifier: Detector):
        super().__init__()
        self.classifier = classifier

    def forward(self, emb: torch.Tensor):
        """
        Args:
            emb: (Batch, Seq_Len, Hidden_Dim)
        Returns:
            start_logits: (Batch, Seq_Len)
            end_logits:   (Batch, Seq_Len, Seq_Len)
        """
        B, N, H = emb.shape
        # FIXME materialize all pairs is expensive.
        start_rep = emb.unsqueeze(2).expand(-1, -1, N, -1)
        end_rep = emb.unsqueeze(1).expand(-1, N, -1, -1)
        pair_emb = torch.cat([start_rep, end_rep], dim=-1)
        logits = self.classifier(pair_emb).squeeze(-1)

        return logits
    

class LitMentionDetector(LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        tokenizer,  #: transformers.PreTrainedTokenizer,
        encoder: torch.nn.Module,
        mention_detector: torch.nn.Module,
        mention_labeler: torch.nn.Module | None = None,
        label2id: dict | None = None,
        lr: float = 2e-5,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "mention_detector", "mention_labeler"])
        self.tokenizer = tokenizer
        self.encoder = encoder
        # Freeze all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mention_detector = mention_detector
        self.mention_labeler = mention_labeler
        self.label2id = label2id
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

        # Two separate metrics for the two tasks
        self.val_f1_start = torchmetrics.classification.BinaryF1Score()
        self.val_f1_end = torchmetrics.classification.BinaryF1Score()
        self.val_f1_mention = torchmetrics.classification.BinaryF1Score()

        if mention_labeler is not None:
            if label2id is None:
                raise ValueError("Need label2id!")
            num_classes = len(self.label2id)
            self.val_f1_entity_start = torchmetrics.classification.BinaryF1Score()
            self.val_f1_entity_end = torchmetrics.classification.BinaryF1Score()
            self.val_f1_entity_mention = torchmetrics.classification.BinaryF1Score()
            self.val_f1_entity_labels = torchmetrics.classification.MulticlassF1Score(
                num_classes=num_classes,
                average="macro"
            )
            self.entity_loss = torch.nn.CrossEntropyLoss()
            log_2 = torch.log(torch.tensor(2.0))
            # TODO Analytical weight to balance losses, but practically who knows.
            self.entity_weight = log_2 / torch.log(torch.tensor(float(num_classes)))

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
            return_offsets_mapping=True,  # needed for word_ids
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        batch_word_ids = []
        for i in range(len(docs)):
            w_ids = [w if w is not None else -1 for w in inputs.word_ids(batch_index=i)]
            batch_word_ids.append(torch.tensor(w_ids))

        word_ids_tensor = torch.stack(batch_word_ids).to(device)
        word_embeddings = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids_tensor
        )
        return word_embeddings
        
    def forward_detector(self, emb: torch.Tensor):
        start_logits, end_logits = self.mention_detector(emb)
        return start_logits, end_logits

    def forward_labeler(self, emb: torch.Tensor):
        entity_logits = self.mention_labeler(emb)
        return entity_logits

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

    def _compute_entity_loss(self, entity_logits, batch):
        """
        entity_logits shape: [batch, max_len, max_len, num_classes]
        """
        preds = []
        targets = []
        
        for b, labels_dict in enumerate(batch["gold_labels"]):
            for (s, e), label_str in labels_dict.items():
                # Ensure indices are within bounds of the current logits
                if s < entity_logits.size(1) and e < entity_logits.size(2):
                    label_id = self.label2id[label_str]
                    # Grab the full vector of class logits [num_classes]
                    preds.append(entity_logits[b, s, e])
                    targets.append(label_id)

        if not targets:
            # Return a zero loss that stays on the correct device and keeps grad_fn
            return entity_logits.sum() * 0

        # Shape: [num_entities, num_classes]
        preds_tensor = torch.stack(preds)
        targets_tensor = torch.tensor(targets, device=entity_logits.device)

        # CrossEntropyLoss handles the mean() internally by default
        return self.entity_loss(preds_tensor, targets_tensor)
    
    def training_step(self, batch, batch_idx):
        emb = self.encode(batch["sentences"])
        start_logits, end_logits = self.forward_detector(emb)
        loss_start = self._compute_start_loss(start_logits, batch)
        loss_end = self._compute_end_loss(end_logits, batch)
        total_loss = loss_start + loss_end
        log_metrics = {
            "train_start_loss": loss_start,
            "train_end_loss": loss_end,
        }
        if batch["task_id"][0] == 1:
            entity_logits = self.forward_labeler(emb)
            loss_entity = self._compute_entity_loss(entity_logits, batch)
            log_metrics["train_entity_loss"] = loss_entity
            total_loss = total_loss + self.entity_weight * loss_entity

        # Final logging
        log_metrics["train_loss"] = total_loss
        self.log_dict(log_metrics, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # 1. SHARED FORWARD PASS
        emb = self.encode(batch["sentences"])
        start_logits, end_logits = self.forward_detector(emb)
        
        token_mask = batch["token_mask"].bool()
        span_loss_mask = batch["span_loss_mask"].bool()
        
        # 2. SHARED EXTRACTION (SIGMOID + THRESHOLD)
        is_start = (torch.sigmoid(start_logits) > self.hparams.threshold).int()
        is_end = (torch.sigmoid(end_logits) > self.hparams.threshold).int()
        
        # Masking logic for valid spans (Upper Triangle + Within Bounds)
        valid_pair_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
        upper_tri = torch.triu(torch.ones_like(end_logits), diagonal=0).bool()
        mention_eval_mask = valid_pair_mask & upper_tri
        
        # Extract flattened predictions and targets
        pred_spans = (is_start.unsqueeze(2) & is_end)[mention_eval_mask]
        target_spans = batch["spans"][mention_eval_mask].int()

        # Dictionary to collect logs for this batch
        log_stats = {}

        # 3. TASK 0: GENERIC MENTIONS
        if batch["task_id"][0] == 0:
            # Safety check: only update if there are actually elements in the masked tensor
            if token_mask.any():
                self.val_f1_start.update(is_start[token_mask], batch["starts"][token_mask].int())
            
            if span_loss_mask.any():
                self.val_f1_end.update(is_end[span_loss_mask], batch["spans"][span_loss_mask].int())
            
            if mention_eval_mask.any():
                self.val_f1_mention.update(pred_spans, target_spans)
                
            log_stats["val_f1_mention"] = self.val_f1_mention

        # 4. TASK 1: ENTITIES
        elif batch["task_id"][0] == 1:
            # Update detector metrics for the entity task
            if token_mask.any():
                self.val_f1_entity_start.update(is_start[token_mask], batch["starts"][token_mask].int())
            
            if span_loss_mask.any():
                self.val_f1_entity_end.update(is_end[span_loss_mask], batch["spans"][span_loss_mask].int())
                
            if mention_eval_mask.any():
                self.val_f1_entity_mention.update(pred_spans, target_spans)
                
            log_stats["val_f1_entity_mention"] = self.val_f1_entity_mention

            # Labeler Classification (on Gold Spans)
            if self.mention_labeler is not None:
                entity_logits = self.forward_labeler(emb)
                gold_preds, gold_targets = [], []
                
                for b, labels_dict in enumerate(batch["gold_labels"]):
                    for (s, e), label_str in labels_dict.items():
                        if s < entity_logits.size(1) and e < entity_logits.size(2):
                            gold_preds.append(torch.argmax(entity_logits[b, s, e], dim=-1))
                            gold_targets.append(self.label2id[label_str])
                
                # Final safety check for the labeler
                if gold_targets:
                    self.val_f1_entity_labels.update(
                        torch.stack(gold_preds), 
                        torch.tensor(gold_targets, device=emb.device)
                    )
                    log_stats["val_f1_entity_labels"] = self.val_f1_entity_labels

        # 5. LOGGING
        # Compute base loss for every batch regardless of task
        loss_start = self._compute_start_loss(start_logits, batch)
        loss_end = self._compute_end_loss(end_logits, batch)
        log_stats["val_loss"] = loss_start + loss_end
        
        self.log_dict(log_stats, prog_bar=True, on_epoch=True, batch_size=len(batch["sentences"]))

    @torch.no_grad()
    def predict_mentions(
        self, sentences: list[list[str]], batch_size: int = 2
    ) -> list[list[tuple[int, int]]]:
        self.eval()
        all_results = []
        thresh = self.hparams.threshold
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            emb = self.encode(batch_sentences)
            start_logits, end_logits = self.forward_detector(emb)
            is_start = torch.sigmoid(start_logits) > thresh
            is_span = torch.sigmoid(end_logits) > thresh
            # Causal j >= i)
            N = end_logits.size(1)
            upper_tri = torch.triu(
                torch.ones((N, N), device=self.device), diagonal=0
            ).bool()
            pred_mask = is_start.unsqueeze(2) & is_span & upper_tri

            # 4. Extract Indices
            indices = pred_mask.nonzero()  # [batch_idx, start_idx, end_idx]

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


@ModelRegistry.register("model_v1")
def make_model_v1(data: DataBlob, model_name="distilroberta-base"):
    dim = 768
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = SentenceEncoder(model_name).train()
    encoder.train()
    start_detector = Detector(dim, dim)
    end_detector = Detector(dim * 2, dim)
    mention_detector = MentionDetectorCore(start_detector, end_detector)
    return LitMentionDetector(tokenizer, encoder, mention_detector)


@ModelRegistry.register("model_v2")
def make_model_v2(data: DataBlob, model_name="distilroberta-base"):
    label2id = data.label2id
    dim = 768
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encoder = SentenceEncoder(model_name).train()
    encoder.train()
    start_detector = Detector(dim, dim)
    end_detector = Detector(dim * 2, dim)
    classifier = Detector(dim * 2, dim, num_classes=len(label2id))
    mention_detector = MentionDetectorCore(start_detector, end_detector)
    mention_labeler = MentionLabeler(classifier)
    return LitMentionDetector(
        tokenizer,
        encoder,
        mention_detector,
        mention_labeler,
        label2id,
    )
