import pytest
import torch
from torch.utils.data import DataLoader
from mentioned.data import (
    mentions_by_sentence,
    flatten_to_sentences,
    LitBankMentionDataset,
    collate_fn,
    make_litbank,
)

# --- FIXTURES ---


@pytest.fixture
def mock_raw_example():
    """Simulates a raw entry from LitBank before flattening."""
    return {
        "sentences": [["The", "cat", "sat", "."], ["It", "was", "happy", "."]],
        "coref_chains": [
            [[0, 0, 1], [1, 0, 0]]  # "The cat" (0,0-1) and "It" (1,0-0)
        ],
    }


@pytest.fixture
def mock_flattened_data():
    """Simulates the output of the HF map functions."""
    return [
        {"sentence": ["The", "cat", "sat", "."], "mentions": [[0, 1]]},
        {"sentence": ["It", "was", "happy", "."], "mentions": [[0, 0]]},
        {"sentence": ["No", "mentions"], "mentions": []},
    ]


# --- UNIT TESTS ---


def test_mentions_by_sentence_grouping(mock_raw_example):
    """Verify coref chains are correctly mapped to sentence indices as strings."""
    result = mentions_by_sentence(mock_raw_example)
    assert "mentions" in result
    # Sentence 0 has mention (0, 1)
    assert (0, 1) in result["mentions"]["0"]
    # Sentence 1 has mention (0, 0)
    assert (0, 0) in result["mentions"]["1"]


def test_flatten_to_sentences_alignment(mock_raw_example):
    """Verify flattening expands 1 doc -> N sentences with correct mention alignment."""
    # Pre-process with mention mapping first
    processed = mentions_by_sentence(mock_raw_example)
    # Mocking HF batch behavior (dict of lists)
    batch = {k: [v] for k, v in processed.items()}

    flattened = flatten_to_sentences(batch)

    assert len(flattened["sentence"]) == 2
    assert flattened["mentions"][0] == [(0, 1)]  # "The cat"
    assert flattened["mentions"][1] == [(0, 0)]  # "It"


def test_dataset_tensor_logic(mock_flattened_data):
    """Verify the 2D span_labels are correctly populated (inclusive indexing)."""
    ds = LitBankMentionDataset(mock_flattened_data)

    # Check sentence with a multi-token mention (0, 1)
    item = ds[0]
    assert item["starts"][0] == 1
    assert item["span_labels"][0, 1] == 1
    assert item["span_labels"].sum() == 1  # Only one mention

    # Check empty sentence
    empty_item = ds[2]
    assert empty_item["starts"].sum() == 0
    assert empty_item["span_labels"].sum() == 0


def test_collate_masking_and_shapes(mock_flattened_data):
    """Verify the 2D mask logic: upper triangle + is_start."""
    ds = LitBankMentionDataset(mock_flattened_data)
    # Batch size 3: [len 4, len 4, len 2]
    batch = [ds[0], ds[1], ds[2]]
    collated = collate_fn(batch)

    B, N = 3, 4
    assert collated["starts"].shape == (B, N)
    assert collated["spans"].shape == (B, N, N)

    # Check span_loss_mask
    # For the first sentence: mention at (0,1). Start is 1 at index 0.
    # Therefore, the mask should allow calculations for row 0.
    mask = collated["span_loss_mask"]

    # Row 0 (starts with 'The') should be mostly True (for j >= 0)
    assert mask[0, 0, 0] == True
    assert mask[0, 0, 1] == True

    # Row 2 (starts with 'sat') should be False because starts[2] == 0
    assert torch.all(mask[0, 2, :] == False)


def test_out_of_bounds_guard():
    """Ensure indexing doesn't crash if data has an error."""
    bad_data = [{"sentence": ["Short"], "mentions": [[0, 10]]}]
    ds = LitBankMentionDataset(bad_data)
    # Should not raise IndexError
    item = ds[0]
    assert item["span_labels"].sum() == 0


# --- INTEGRATION TEST ---
def test_make_litbank_integration():
    """Check if the real pipeline loads and provides a valid batch."""
    try:
        train_loader, _, _ = make_litbank(tag="split_0")
        batch = next(iter(train_loader))

        assert "sentences" in batch
        assert "span_loss_mask" in batch
        assert isinstance(batch["spans"], torch.Tensor)
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")
