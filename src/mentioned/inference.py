import os
import onnxruntime as ort

import torch
import torch.nn as nn
import numpy as np

from mentioned.model import ModelRegistry, LitMentionDetector
import onnxruntime as ort


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


class ONNXMentionDetectorPipeline:
    def __init__(self, model_path: str, tokenizer, threshold: float = 0.5):
        # 1. Load the ONNX session
        # 'CPUExecutionProvider' is perfect for HF Space free tier
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = tokenizer
        # We still use your existing MentionProcessor for the tokenization math
        self.processor = MentionProcessor(tokenizer)
        self.threshold = threshold

    def predict(self, docs: list[list[str]]):
        batch = self.processor(docs)
        onnx_inputs = {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
            "word_ids": batch["word_ids"].numpy(),
        }
        start_probs, end_probs = self.session.run(None, onnx_inputs)

        # 5. Extraction Logic (Numpy version)
        results = []
        for i in range(len(docs)):
            doc_mentions = []
            doc_len = len(docs[i])
            # Slice to actual word length and apply threshold
            is_start = start_probs[i, :doc_len] > self.threshold
            is_span = end_probs[i, :doc_len, :doc_len] > self.threshold
            # Causal mask (j >= i) using numpy
            upper_tri = np.triu(np.ones((doc_len, doc_len), dtype=bool))
            # Find indices where (start is true) AND (span is true) AND (end >= start)
            combined_mask = is_span & is_start[:, None] & upper_tri
            final_indices = np.argwhere(combined_mask)

            for s_idx, e_idx in final_indices:
                score = end_probs[i, s_idx, e_idx]
                doc_mentions.append(
                    {
                        "start": int(s_idx),
                        "end": int(e_idx),
                        "score": round(float(score), 4),
                        "text": " ".join(docs[i][s_idx : e_idx + 1]),
                    }
                )
            results.append(doc_mentions)

        return results


def create_inference_model(
    repo_id: str,
    model_factory: str,
    device: str = "cpu",
):
    """
    Factory to load a trained model from HF Hub and wrap it for ONNX/Inference.
    """
    # 1. Load the Lightning model (with its weights)
    # Note: Ensure LitMentionDetector is defined in your scope
    fresh_model = ModelRegistry.get(model_factory)()
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


def compile_inference_model(model, output_dir="model_v1_onnx"):
    """ONNX export with dynamic axes for."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    model.tokenizer.save_pretrained(output_dir)
    onnx_path = os.path.join(output_dir, "model.onnx")
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "word_ids": {0: "batch", 1: "sequence"},
        "start_probs": {0: "batch", 1: "num_words"},
        "end_probs": {0: "batch", 1: "num_words", 2: "num_words"},
    }

    # Dummy inputs as before
    dummy_inputs = (
        torch.randint(0, 100, (1, 16), dtype=torch.long),
        torch.ones((1, 16), dtype=torch.long),
        torch.arange(16, dtype=torch.long).unsqueeze(0),
    )

    print("🚀 Re-exporting with legacy engine (dynamo=False)...")

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        export_params=True,
        opset_version=17,  # Use 17 for maximum compatibility with legacy mode
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "word_ids"],
        output_names=["start_probs", "end_probs"],
        dynamic_axes=dynamic_axes,
        dynamo=False,  # <--- FORCE THE OLD, STABLE EXPORTER
    )
    print(f"✅ Exported to {output_dir}! Checking dimensions...")

    # Verification:
    sess = ort.InferenceSession(onnx_path)
    for input_meta in sess.get_inputs():
        print(f"Input '{input_meta.name}' shape: {input_meta.shape}")


if __name__ == "__main__":
    repo_id = "kadarakos/mention-detector-poc-dry-run"
    model_factory = "model_v1"
    inference_model_path = "model_v1_onnx"
    inference_model = create_inference_model(repo_id, model_factory)
    compile_inference_model(inference_model, inference_model_path)
    pipeline = ONNXMentionDetectorPipeline(
        model_path=os.path.join(inference_model_path, "model.onnx"),
        tokenizer=inference_model.tokenizer,
        # XXX Sweet spot for this examples on this model, found by hand
        threshold=0.3,
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
        print(" ".join(docs[i]))
        preds = []
        for mention in mentions:
            preds.append(mention["text"])
        print(preds)
