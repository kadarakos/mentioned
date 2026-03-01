import pytest
import torch
import numpy as np
import os
from unittest.mock import MagicMock, patch

# IMPORTANT: Replace 'mentioned.inference' with your actual filename/package
from mentioned.inference import (
    InferenceMentionDetector,
    MentionProcessor,
    ONNXMentionDetectorPipeline,
)

# --- FIXTURES ---


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()

    # 1. Mock the BatchEncoding object returned by calling the tokenizer
    mock_encoding = MagicMock()
    mock_encoding.__getitem__.side_effect = {
        "input_ids": torch.tensor([[101, 102, 103, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1]]),
    }.get

    # 2. Mock the .word_ids() method specifically
    mock_encoding.word_ids.return_value = [None, 0, 1, None]

    tokenizer.return_value = mock_encoding
    return tokenizer


@pytest.fixture
def mock_inference_detector():
    encoder = MagicMock(spec=torch.nn.Module)
    encoder.max_length = 512
    encoder.return_value = torch.randn(1, 4, 128)

    mention_det = MagicMock(spec=torch.nn.Module)
    mention_det.return_value = (torch.randn(1, 2), torch.randn(1, 2, 2))

    return InferenceMentionDetector(encoder, mention_det)


# --- TESTS ---


def test_mention_processor_word_id_mapping(mock_tokenizer):
    processor = MentionProcessor(mock_tokenizer, max_length=10)
    docs = [["The", "cat"]]

    batch = processor(docs)

    assert "word_ids" in batch
    # [None, 0, 1, None] -> [-1, 0, 1, -1]
    expected = torch.tensor([[-1, 0, 1, -1]])
    assert torch.equal(batch["word_ids"], expected)


def test_pipeline_extraction_logic():
    """Verify Numpy extraction: thresholding and causal masking."""

    # Prevent ONNX from trying to load a file from disk during init
    with patch("onnxruntime.InferenceSession") as mock_session_class:
        mock_session_instance = mock_session_class.return_value

        # 1 doc, 3 words: "The", "big", "cat"
        s_probs = np.array([[0.9, 0.1, 0.1]])
        e_probs = np.array([[[0.1, 0.1, 0.9], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]])

        mock_session_instance.run.return_value = [s_probs, e_probs]

        tokenizer = MagicMock()
        pipeline = ONNXMentionDetectorPipeline("dummy.onnx", tokenizer, threshold=0.5)

        # Mock the processor so it doesn't actually call a real tokenizer
        pipeline.processor = MagicMock(
            return_value={
                "input_ids": torch.zeros((1, 5)),
                "attention_mask": torch.zeros((1, 5)),
                "word_ids": torch.zeros((1, 5)),
            }
        )

        docs = [["The", "big", "cat"]]
        results = pipeline.predict(docs)

        assert len(results) == 1
        mention = results[0][0]  # First doc, first mention
        assert mention["start"] == 0
        assert mention["end"] == 2
        assert mention["text"] == "The big cat"
        assert mention["score"] == 0.9


def test_onnx_export_compilation(mock_inference_detector, tmp_path):
    """Verify that the model can be exported via torch.onnx.export."""
    # We must import the function from your specific file
    from mentioned.inference import compile_inference_model

    mock_inference_detector.tokenizer = MagicMock()
    output_dir = tmp_path / "onnx_test"

    # Use a try-except to get better error visibility during export
    try:
        compile_inference_model(mock_inference_detector, output_dir=str(output_dir))
    except Exception as e:
        pytest.fail(f"ONNX Export failed: {e}")

    assert os.path.exists(output_dir / "model.onnx")
