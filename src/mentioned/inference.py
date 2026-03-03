import os
import onnxruntime as ort

import torch
import torch.nn as nn
import numpy as np

from mentioned.model import ModelRegistry, LitMentionDetector
from mentioned.data import DataRegistry


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


class InferenceMentionLabeler(nn.Module):
    def __init__(self, encoder, mention_detector, mention_labeler, id2label):
        super().__init__()
        self.encoder = encoder
        self.mention_detector = mention_detector
        self.mention_labeler = mention_labeler
        self.id2label = id2label

    def forward(self, input_ids, attention_mask, word_ids):
        """
        Pure tensor forward pass for ONNX export.

        Returns (Tensors):
            start_probs: (B, N)
            end_probs: (B, N, N)
            label_probs: (B, N, N, C) or dummy empty tensor
        """
        # 1. Encoder pass
        word_embeddings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_ids=word_ids
        )
        start_logits, end_logits = self.mention_detector(word_embeddings)
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)
        entity_logits = self.mention_labeler(word_embeddings)
        label_probs = torch.softmax(entity_logits, dim=-1)
        return start_probs, end_probs, label_probs


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
            model_path,
            providers=['CPUExecutionProvider']
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
            "word_ids": batch["word_ids"].numpy()
        }
        start_probs, end_probs = self.session.run(None, onnx_inputs)

        # 5. Extraction Logic (Numpy version)
        results = []
        for i in range(len(docs)):
            doc_mentions = []
            doc_len = len(docs[i])
            is_start = start_probs[i, :doc_len] > self.threshold
            is_span = end_probs[i, :doc_len, :doc_len] > self.threshold
            upper_tri = np.triu(np.ones((doc_len, doc_len), dtype=bool))
            combined_mask = is_span & is_start[:, None] & upper_tri
            final_indices = np.argwhere(combined_mask)

            for s_idx, e_idx in final_indices:
                # XXX Considering end-prob as mention score!
                score = end_probs[i, s_idx, e_idx]
                doc_mentions.append({
                    "start": int(s_idx),
                    "end": int(e_idx),
                    "score": round(float(score), 4),
                    "text": " ".join(docs[i][s_idx:e_idx + 1]),
                })
            results.append(doc_mentions)

        return results


class ONNXMentionLabelerPipeline:
    def __init__(self, model_path: str, tokenizer, id2label: dict = None, threshold: float = 0.5):
        # 1. Load the ONNX session
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.tokenizer = tokenizer
        self.processor = MentionProcessor(tokenizer)
        self.threshold = threshold
        # Mapping for human-readable labels
        self.id2label = id2label

    def predict(self, docs: list[list[str]]):
        batch = self.processor(docs)
        onnx_inputs = {
            "input_ids": batch["input_ids"].numpy(),
            "attention_mask": batch["attention_mask"].numpy(),
            "word_ids": batch["word_ids"].numpy()
        }
        
        # ONNX returns a list of outputs in order: [start_probs, end_probs, label_probs]
        start_probs, end_probs, label_probs = self.session.run(None, onnx_inputs)

        results = []
        for i in range(len(docs)):
            doc_mentions = []
            doc_len = len(docs[i])
            is_start = start_probs[i, :doc_len] > self.threshold
            is_span = end_probs[i, :doc_len, :doc_len] > self.threshold
            upper_tri = np.triu(np.ones((doc_len, doc_len), dtype=bool))
            combined_mask = is_span & is_start[:, None] & upper_tri
            final_indices = np.argwhere(combined_mask)

            for s_idx, e_idx in final_indices:
                # XXX Considering end-prob as score!
                det_score = float(end_probs[i, s_idx, e_idx])
                class_probs = label_probs[i, s_idx, e_idx]
                label_id = int(np.argmax(class_probs))
                label_score = float(class_probs[label_id])
                
                doc_mentions.append({
                    "start": int(s_idx),
                    "end": int(e_idx),
                    "text": " ".join(docs[i][s_idx : e_idx + 1]),
                    "score": round(det_score, 4),
                    "label": self.id2label.get(label_id, str(label_id)),
                    "label_score": round(label_score, 4),
                })
            results.append(doc_mentions)

        return results


def create_inference_model(
    repo_id: str,
    encoder_id: str,
    model_factory: str,
    data_factory: str,
    device: str = "cpu",
):
    """
    Factory to load a trained model from HF Hub and wrap it for ONNX/Inference.
    """
    data = DataRegistry.get(data_factory)()
    fresh_bundle = ModelRegistry.get(model_factory)(data, encoder_id)
    labeler = getattr(fresh_bundle, "mention_labeler", None)
    l2id = getattr(fresh_bundle, "label2id", None)

    lit_model = LitMentionDetector.from_pretrained(
        repo_id,
        tokenizer=fresh_bundle.tokenizer,
        encoder=fresh_bundle.encoder,
        mention_detector=fresh_bundle.mention_detector,
        label2id=l2id,
        mention_labeler=labeler,
        # weights_only=False,
    )
    lit_model.to(device)
    lit_model.eval()
    if l2id is not None:
        id2l = {v: k for k, v in l2id.items()}
        inference_model = InferenceMentionLabeler(
            encoder=lit_model.encoder,
            mention_detector=lit_model.mention_detector,
            mention_labeler=lit_model.mention_labeler,
            id2label=id2l,
        )
    else:
        inference_model = InferenceMentionDetector(
            encoder=lit_model.encoder, mention_detector=lit_model.mention_detector
        )
    inference_model.tokenizer = lit_model.tokenizer
    inference_model.max_length = lit_model.encoder.max_length

    return inference_model.eval()


def compile_detector(model, output_dir="model_v1_onnx"):
    """ONNX export with dynamic axes for."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    model.tokenizer.save_pretrained(output_dir)
    onnx_path = os.path.join(output_dir, "model.onnx")
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "word_ids":       {0: "batch", 1: "sequence"},
        "start_probs":    {0: "batch", 1: "num_words"},
        "end_probs":      {0: "batch", 1: "num_words", 2: "num_words"}
    }

    # Dummy inputs as before
    dummy_inputs = (
        torch.randint(0, 100, (1, 16), dtype=torch.long),
        torch.ones((1, 16), dtype=torch.long),
        torch.arange(16, dtype=torch.long).unsqueeze(0)
    )

    print("🚀 Re-exporting with legacy engine (dynamo=False)...")

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        export_params=True,
        opset_version=17, # Use 17 for maximum compatibility with legacy mode
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask", "word_ids"],
        output_names=["start_probs", "end_probs"],
        dynamic_axes=dynamic_axes,
        dynamo=False  # <--- FORCE THE OLD, STABLE EXPORTER
    )
    print(f"✅ Exported to {output_dir}! Checking dimensions...")

    # Verification:
    sess = ort.InferenceSession(onnx_path)
    for input_meta in sess.get_inputs():
        print(f"Input '{input_meta.name}' shape: {input_meta.shape}")


def compile_labeler(model, output_dir="labeler_onnx"):
    model.cpu().eval() 
    os.makedirs(output_dir, exist_ok=True)
    model.tokenizer.save_pretrained(output_dir)
    onnx_path = os.path.join(output_dir, "model.onnx")

    print(f"🛠️ Exporting {model.__class__.__name__} to {onnx_path}...")

    # Realistic dummy inputs: 2 batches, 16 tokens
    dummy_inputs = (
        torch.randint(0, 50000, (2, 16), dtype=torch.long),
        torch.ones((2, 16), dtype=torch.long),
        torch.arange(16, dtype=torch.long).unsqueeze(0).repeat(2, 1)
    )

    # Rename sequence axes to unique names to stop the merging warning
    dynamic_axes = {
        "input_ids":      {0: "batch", 1: "seq_ids"},
        "attention_mask": {0: "batch", 1: "seq_mask"},
        "word_ids":       {0: "batch", 1: "seq_words"},
        "start_probs":    {0: "batch", 1: "num_words"},
        "end_probs":      {0: "batch", 1: "num_words", 2: "num_words"},
        "label_probs":    {0: "batch", 1: "num_words", 2: "num_words", 3: "num_classes"}
    }

    torch.onnx.export(
        model,
        dummy_inputs,
        onnx_path,
        export_params=True,
        opset_version=17, # Use 17 for better support of modern ops
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask', 'word_ids'],
        output_names=['start_probs', 'end_probs', 'label_probs'],
        dynamic_axes=dynamic_axes,
        # THE FIXES:
        training=torch.onnx.TrainingMode.EVAL,
        dynamo=False 
    )
    print("✅ Export finished successfully!")


if __name__ == "__main__":
    # repo_id = "kadarakos/mention-detector-poc-dry-run"
    # model_factory = "model_v1"
    # inference_model_path = "model_v1_onnx"
    model_factory = "model_v2"
    data_factory = "litbank_entities"
    inference_model_path = "model_v2_onnx"
    repo_id = "kadarakos/entity-labeler-poc"
    encoder_id = "distilroberta-base"
    inference_model = create_inference_model(
        repo_id,
        encoder_id,
        model_factory,
        data_factory,
    )
    if isinstance(inference_model, InferenceMentionDetector):
        compile_detector(inference_model, inference_model_path)
        pipeline = ONNXMentionDetectorPipeline(
            model_path=os.path.join(inference_model_path, "model.onnx"),
            tokenizer=inference_model.tokenizer,
            # XXX Sweet spot for this examples on this model, found by hand
            threshold=0.3,
        )
    else:
        print(inference_model)
        compile_labeler(inference_model, inference_model_path)
        pipeline = ONNXMentionLabelerPipeline(
            model_path=os.path.join(inference_model_path, "model.onnx"),
            tokenizer=inference_model.tokenizer,
            threshold=0.5,
            id2label=inference_model.id2label,
        )
    print("FUCK")
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
            preds.append((mention["text"], mention["label"]))
        print(preds)
