import torch
import torch.nn as nn

from mentioned.model import make_model_v1, LitMentionDetector


class InferenceMentionDetector(nn.Module):
    def __init__(self, encoder, mention_detector):
        super().__init__()
        self.encoder = encoder
        self.mention_detector = mention_detector

    def forward(self, input_ids, attention_mask, word_ids):
        """
        Inputs (Tensors):
            input_ids: (B, Seq_Len)
            attention_mask: (B, Seq_Len)
            word_ids: (B, Seq_Len) -> Word index per token, -1 padding.

        Returns (Tensors):
            start_probs: (B, Num_Words)
            end_probs: (B, Num_Words, Num_Words)
        """
        # B x N x D
        word_embeddings = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, word_ids=word_ids
        )
        # B x N, B x N x N
        start_logits, end_logits = self.mention_detector(word_embeddings)
        # B x N
        start_probs = torch.sigmoid(start_logits)
        # B x N x N
        end_probs = torch.sigmoid(end_logits)

        return start_probs, end_probs


class MentionProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, docs: list[list[str]]):
        """
        Converts raw word lists into tensors.
        Args:
            docs: List of documents, where each doc is a list of words.
                  Example: [["Hello", "world"], ["Testing", "this"]]
        """
        inputs = self.tokenizer(
            docs,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_attention_mask=True,
        )

        # We need a tensor where each token index maps to its word index.
        # Special tokens (<s>, </s>, <pad>) are mapped to -1.
        batch_word_ids = []
        for i in range(len(docs)):
            # tokenizer.word_ids(i) returns [None, 0, 1, 1, 2, None]
            w_ids = [w if w is not None else -1 for w in inputs.word_ids(batch_index=i)]
            batch_word_ids.append(torch.tensor(w_ids))
        word_ids_tensor = torch.stack(batch_word_ids)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "word_ids": word_ids_tensor,
        }


class MentionDetectorPipeline:
    def __init__(
        self,
        model: LitMentionDetector,
        tokenizer,
        threshold: float = 0.5,
    ):
        """
        Args:
            model: The LitMentionDetector (LightningModule)
            tokenizer: The PreTrainedTokenizer
            threshold: Probability threshold for both start and span filters
        """
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.processor = MentionProcessor(tokenizer, model.max_length)
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, docs: list[list[str]]):
        """
        Args:
            docs: List of documents (each is a list of words)
        Returns:
            List of lists containing dicts: {"start": int, "end": int, "score": float, "text": str}
        """
        batch = self.processor(docs)
        # Move batch to the same device as the model
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        start_probs, end_probs = self.model(**batch)

        # Extract mentions.
        results = []
        for i in range(len(docs)):
            doc_mentions = []
            doc_len = len(docs[i])
            is_start = start_probs[i, :doc_len] > self.threshold
            is_span = end_probs[i, :doc_len, :doc_len] > self.threshold
            # Causal mask.
            upper_tri = torch.triu(
                torch.ones((doc_len, doc_len), device=self.device), diagonal=0
            ).bool()
            # Full mask.
            combined_mask = is_span & is_start.unsqueeze(1) & upper_tri
            final_indices = combined_mask.nonzero()

            for span in final_indices:
                s_idx, e_idx = span[0].item(), span[1].item()
                # End prob is interpreted as confidence score.
                score = end_probs[i, s_idx, e_idx].item()
                doc_mentions.append(
                    {
                        "start": s_idx,
                        "end": e_idx,
                        "score": round(score, 4),
                        "text": " ".join(docs[i][s_idx : e_idx + 1]),
                    }
                )

            results.append(doc_mentions)

        return results


def create_inference_model(repo_id: str, device: str = "cpu"):
    """
    Factory to load a trained model from HF Hub and wrap it for ONNX/Inference.
    """
    # 1. Load the Lightning model (with its weights)
    # Note: Ensure LitMentionDetector is defined in your scope
    fresh_model = make_model_v1()
    lit_model = LitMentionDetector.from_pretrained(
        repo_id,
        tokenizer=fresh_model.tokenizer,
        encoder=fresh_model.encoder,
        mention_detector=fresh_model.mention_detector,
    )

    # 2. Move to device and set to eval mode
    lit_model.to(device)
    lit_model.eval()

    # 3. Wrap the core components into the Inference class
    # This separates the 'training' logic from the 'inference' graph
    inference_model = InferenceMentionDetector(
        encoder=lit_model.encoder, mention_detector=lit_model.mention_detector
    )

    # 4. Attach the tokenizer and max_length for the Preprocessor
    # (Optional: helpful for keeping metadata together)
    inference_model.tokenizer = lit_model.tokenizer
    inference_model.max_length = lit_model.encoder.max_length

    return inference_model.eval()


# TODO
def compile_inference_model(model):
    return model


repo_id = "kadarakos/mention-detector-poc-dry-run"
inference_model = compile_inference_model(create_inference_model(repo_id))
pipeline = MentionDetectorPipeline(
    model=inference_model,
    tokenizer=inference_model.tokenizer,
    threshold=0.6,  # Noticed that precision is bad below this (still bad :D).
)

docs = [
    "Does this model actually work?".split(),
    "The name of the mage is Bubba.".split(),
    "It was quite a sunny day when the model finally started working.".split(),
    "Albert Einstein was a theoretical physicist who developed the theory of relativity".split(),
    "Apple Inc. and Microsoft are competing in the cloud computing market".split(),
    "New York City is often called the Big Apple".split(),
    "The Great Barrier Reef is the world's largest coral reef system".split(),
    "Marie Curie was the first woman to win a Nobel Prize".split(),
]

batch_mentions = pipeline.predict(docs)
for i, mentions in enumerate(batch_mentions):
    print(docs[i])
    for mention in mentions:
        print(mention["text"])
