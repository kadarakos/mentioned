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
            word_ids: (B, Seq_Len) -> Word index per token, -1 for special tokens

        Returns (Tensors):
            start_probs: (B, Num_Words)
            end_probs: (B, Num_Words, Num_Words)
        """

        # 1. Subword-to-Word Pooling (The vectorized logic we wrote earlier)
        # Returns: (Batch, Num_Words, Hidden_Dim)
        word_embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids
        )

        # 2. Mention Detection Logic
        # Returns: start_logits (B, W), end_logits (B, W, W)
        start_logits, end_logits = self.mention_detector(word_embeddings)

        # 3. Probabilities for Inference
        # Applying sigmoid here makes the ONNX model output final scores
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)

        return start_probs, end_probs


class MentionProcessor:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, docs: list[list[str]]):
        """
        Converts raw word lists into tensors for the ONNX model.
        Args:
            docs: List of documents, where each doc is a list of words.
                  Example: [["Hello", "world"], ["Testing", "this"]]
        """
        # 1. Standard Tokenization
        # is_split_into_words=True is crucial since your input is list[list[str]]
        inputs = self.tokenizer(
            docs,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_attention_mask=True
        )

        # 2. Map Subwords to Word IDs
        # We need a tensor where each token index maps to its word index.
        # Special tokens (<s>, </s>, <pad>) are mapped to -1 to be ignored by pooling.
        batch_word_ids = []
        for i in range(len(docs)):
            # tokenizer.word_ids(i) returns [None, 0, 1, 1, 2, None]
            w_ids = [w if w is not None else -1 for w in inputs.word_ids(batch_index=i)]
            batch_word_ids.append(torch.tensor(w_ids))

        # 3. Stack into a batch tensor (Batch, Seq_Len)
        word_ids_tensor = torch.stack(batch_word_ids)

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "word_ids": word_ids_tensor
        }


class MentionDetectorPipeline:
    def __init__(self, model, tokenizer, threshold: float = 0.5):
        """
        Args:
            model: The InferenceMentionDetector (PyTorch or ONNX Session)
            tokenizer: The PreTrainedTokenizer
            threshold: Probability threshold to consider a mention valid
        """
        self.model = model.eval()
        self.processor = MentionProcessor(tokenizer, model.max_length)
        self.threshold = threshold

    @torch.no_grad()
    def predict(self, docs: list[list[str]]):
        """
        Args:
            docs: List of documents (each is a list of words)
        Returns:
            List of lists containing dicts: {"start": int, "end": int, "score": float}
        """
        # 1. Preprocess to Tensors
        batch = self.processor(docs)
        device = next(self.model.parameters()).device

        # Move batch to model device
        batch = {k: v.to(device) for k, v in batch.items()}

        # 2. Forward Pass
        # start_probs: (B, W), end_probs: (B, W, W)
        start_probs, end_probs = self.model(**batch)

        # 3. Post-process: Extract Mentions
        results = []
        for i in range(len(docs)):
            doc_mentions = []
            doc_len = len(docs[i])

            # We only look at the valid word range for this specific document
            # end_probs[i] is a (W, W) matrix where [s, e] is the prob of span s->e
            valid_spans = (end_probs[i][:doc_len, :doc_len] > self.threshold).nonzero()

            for span in valid_spans:
                start_idx = span[0].item()
                end_idx = span[1].item()

                # Logic: Only valid if end >= start
                if end_idx >= start_idx:
                    score = end_probs[i, start_idx, end_idx].item()
                    doc_mentions.append({
                        "start": start_idx,
                        "end": end_idx,
                        "score": round(score, 4),
                        "text": " ".join(docs[i][start_idx : end_idx + 1])
                    })

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
        encoder=lit_model.encoder,
        mention_detector=lit_model.mention_detector
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
inference_model = compile_inference_model(
    create_inference_model(repo_id)
)
pipeline = MentionDetectorPipeline(
    model=inference_model,
    tokenizer=inference_model.tokenizer,
    threshold=0.6  # Noticed that precision is bad below this (still bad :D).
)

docs = [
    "Does this model actually work?".split(),
    "The name of the mage is Bubba.".split(),
    "It was quite a sunny day when the model finally started working.".split(),
    "Albert Einstein was a theoretical physicist who developed the theory of relativity".split(),
    "Apple Inc. and Microsoft are competing in the cloud computing market".split(),
    "New York City is often called the Big Apple".split(),
    "The Great Barrier Reef is the world's largest coral reef system".split(),
    "Marie Curie was the first woman to win a Nobel Prize".split()
]

batch_mentions = pipeline.predict(docs)
for i, mentions in enumerate(batch_mentions):
    print(docs[i])
    for mention in mentions:
        print(mention["text"])
