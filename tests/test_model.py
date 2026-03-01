import pytest
import torch
from transformers import AutoConfig
from mentioned.model import SentenceEncoder, Detector, MentionDetectorCore

# --- FIXTURES ---


@pytest.fixture
def model_dims():
    return {"input_dim": 128, "hidden_dim": 64, "seq_len": 10, "batch_size": 2}


@pytest.fixture
def mock_embeddings(model_dims):
    # Create the tensor and immediately enable gradient tracking
    return torch.randn(
        model_dims["batch_size"],
        model_dims["seq_len"],
        model_dims["input_dim"],
        requires_grad=True,  # <--- CRITICAL FIX
    )


@pytest.mark.parametrize("subwords_per_word", [1, 2, 4])
def test_variable_subword_pooling(subwords_per_word):
    encoder = SentenceEncoder(model_name="sshleifer/tiny-distilroberta-base")
    B, Hidden = 1, encoder.dim
    Total_Subwords = 8
    Num_Words = Total_Subwords // subwords_per_word

    input_ids = torch.randint(0, 100, (B, Total_Subwords))
    attention_mask = torch.ones(B, Total_Subwords)

    # Create word_ids like [0, 0, 1, 1, 2, 2, 3, 3]
    word_ids = torch.arange(Num_Words).repeat_interleave(subwords_per_word).unsqueeze(0)

    word_embs = encoder(input_ids, attention_mask, word_ids)

    assert word_embs.shape == (B, Num_Words, Hidden)
    assert not torch.isnan(word_embs).any()


def test_detector_projections(model_dims, mock_embeddings):
    """Verify Detector handles both 3D (starts) and 4D (spans) tensors."""
    detector = Detector(model_dims["input_dim"], model_dims["hidden_dim"])

    # Test 3D (Start Logits)
    start_out = detector(mock_embeddings)
    assert start_out.shape == (model_dims["batch_size"], model_dims["seq_len"], 1)

    # Test 4D (Pair Logits)
    pair_input = torch.randn(2, 10, 10, model_dims["input_dim"])
    pair_out = detector(pair_input)
    assert pair_out.shape == (2, 10, 10, 1)


def test_mention_detector_core_logic(model_dims, mock_embeddings):
    """Verify the N x N pair materialization and concatenation."""
    B, N, H = model_dims["batch_size"], model_dims["seq_len"], model_dims["input_dim"]

    start_det = Detector(H, 32)
    # End detector expects concat of 2 embeddings (H + H)
    end_det = Detector(H * 2, 32)

    model = MentionDetectorCore(start_det, end_det)
    start_logits, end_logits = model(mock_embeddings)

    assert start_logits.shape == (B, N)
    assert end_logits.shape == (B, N, N)

    # Ensure gradients can flow back to embeddings
    loss = start_logits.sum() + end_logits.sum()
    loss.backward()
    assert mock_embeddings.grad is not None
