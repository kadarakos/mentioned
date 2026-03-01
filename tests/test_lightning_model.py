import pytest
import torch
from unittest.mock import MagicMock
from mentioned.model import LitMentionDetector, Detector, MentionDetectorCore

# --- FIXTURES ---


@pytest.fixture
def mock_components():
    """Create minimal components for the LightningModule."""
    # Use a tiny mock encoder or a small transformer config
    encoder = MagicMock(spec=torch.nn.Module)
    encoder.max_length = 512
    encoder.dim = 128
    encoder.parameters.return_value = [torch.nn.Parameter(torch.randn(128, 128))]

    start_det = Detector(128, 64)
    end_det = Detector(256, 64)
    mention_det = MentionDetectorCore(start_det, end_det)

    tokenizer = MagicMock()
    # Mock tokenizer output for 'encode' method
    tokenizer.return_value = {
        "input_ids": torch.zeros((1, 5)),
        "attention_mask": torch.ones((1, 5)),
    }
    # Mock word_ids method
    tokenizer.word_ids.return_value = [0, 0, 1, 1, None]

    return tokenizer, encoder, mention_det


@pytest.fixture
def dummy_batch():
    """A realistic batch as produced by your collate_fn."""
    B, N = 2, 4
    return {
        "sentences": [["Hello", "world"], ["Test"]],
        "starts": torch.zeros((B, N), dtype=torch.long),
        "spans": torch.zeros((B, N, N), dtype=torch.long),
        "token_mask": torch.ones((B, N), dtype=torch.bool),
        "span_loss_mask": torch.ones((B, N, N), dtype=torch.bool),
    }


# --- TESTS ---


def test_encoder_freezing(mock_components):
    """Verify the encoder parameters are set to requires_grad=False."""
    tokenizer, encoder, mention_det = mock_components
    model = LitMentionDetector(tokenizer, encoder, mention_det)

    for param in model.encoder.parameters():
        assert param.requires_grad is False

    # Ensure detector remains trainable
    for param in model.mention_detector.parameters():
        assert param.requires_grad is True


def test_compute_loss_empty_mask(mock_components, dummy_batch):
    tokenizer, encoder, mention_det = mock_components
    model = LitMentionDetector(tokenizer, encoder, mention_det)

    # Force a zeroed out mask
    dummy_batch["span_loss_mask"] = torch.zeros_like(
        dummy_batch["span_loss_mask"]
    ).bool()

    # IMPORTANT: Logits must require grad for the loss to require grad
    end_logits = torch.randn(2, 4, 4, requires_grad=True)

    loss = model._compute_end_loss(end_logits, dummy_batch)

    assert loss.item() == 0.0
    # This will now pass because end_logits.sum() * 0 preserves the grad_fn
    assert loss.requires_grad


def test_predict_mentions_format(mock_components):
    """Verify predict_mentions returns the expected list of (start, end) tuples."""
    tokenizer, encoder, mention_det = mock_components
    model = LitMentionDetector(tokenizer, encoder, mention_det)

    # Mock the internal encode/forward to avoid real transformer calls
    model.encode = MagicMock(return_value=torch.randn(1, 3, 128))

    # Mock logits to trigger exactly one mention at (0, 1)
    # start_logits (B, N), end_logits (B, N, N)
    s_logits = torch.tensor([[10.0, -10.0, -10.0]])  # Index 0 is a start
    e_logits = torch.tensor(
        [[[-10.0, 10.0, -10.0], [-10.0, -10.0, -10.0], [-10.0, -10.0, -10.0]]]
    )  # (0,1) is a span

    model.mention_detector.forward = MagicMock(return_value=(s_logits, e_logits))

    results = model.predict_mentions([["Mock", "sentence", "."]])

    assert len(results) == 1
    assert results[0] == [(0, 1)]


def test_training_step_integration(mock_components, dummy_batch):
    """Verify training_step returns a valid loss tensor."""
    tokenizer, encoder, mention_det = mock_components
    model = LitMentionDetector(tokenizer, encoder, mention_det)

    # Mock encode to return a fixed tensor
    model.encode = MagicMock(return_value=torch.randn(2, 4, 128))

    loss = model.training_step(dummy_batch, 0)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_validation_metrics_update(mock_components, dummy_batch):
    """Ensure F1 metrics are updated during validation."""
    tokenizer, encoder, mention_det = mock_components
    model = LitMentionDetector(tokenizer, encoder, mention_det)
    model.encode = MagicMock(return_value=torch.randn(2, 4, 128))

    # Populate metrics
    model.validation_step(dummy_batch, 0)

    # Check if metrics were touched
    assert model.val_f1_start.update_count > 0
    assert model.val_f1_mention.update_count > 0
