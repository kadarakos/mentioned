import pytest
import torch
from torch.utils.data import DataLoader
from unittest.mock import patch
from mentioned.data import (
    mentions_by_sentence,
    flatten_to_sentences,
    LitBankMentionDataset,
    collate_fn,
    make_litbank,
    extract_spans_from_bio,
    flatten_entities,
    LitBankEntityDataset,
    entity_collate_fn,
    make_litbank_entity,
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

def test_extract_spans_from_bio_simple():
    sentence = [
        {"token": "John", "bio_tags": ["B-PER"]},
        {"token": "Smith", "bio_tags": ["I-PER"]},
        {"token": "works", "bio_tags": ["O"]},
        {"token": "at", "bio_tags": ["O"]},
        {"token": "Google", "bio_tags": ["B-ORG"]},
    ]

    spans, labels = extract_spans_from_bio(sentence)

    # inclusive indexing
    assert spans == [(0, 1), (4, 4)]
    assert labels == ["PER", "ORG"]


def test_extract_spans_handles_single_token_entity():
    sentence = [
        {"token": "Paris", "bio_tags": ["B-LOC"]},
        {"token": "is", "bio_tags": ["O"]},
    ]

    spans, labels = extract_spans_from_bio(sentence)

    assert spans == [(0, 0)]
    assert labels == ["LOC"]


def test_litbank_entity_dataset_getitem():
    fake_dataset = [
        {
            "sentence": ["John", "works"],
            "entity_spans": [(0, 1)],
            "entity_labels": ["PER"],
        }
    ]

    ds = LitBankEntityDataset(fake_dataset)
    item = ds[0]

    assert item["sentence"] == ["John", "works"]
    assert torch.equal(item["starts"], torch.tensor([1, 0]))
    assert item["entity_spans"] == [(0, 1)]
    assert item["entity_labels"] == ["PER"]
    assert item["task_id"] == 1



def test_flatten_entities():
    batch = {
        "entities": [
            [  # document 1
                [
                    {"token": "John", "bio_tags": ["B-PER"]},
                    {"token": "Smith", "bio_tags": ["I-PER"]},
                ]
            ]
        ]
    }

    output = flatten_entities(batch)

    assert output["sentence"] == [["John", "Smith"]]
    assert output["entity_spans"] == [[(0, 1)]]
    assert output["entity_labels"] == [["PER"]]


def test_entity_collate_fn_basic():
    batch = [
        {
            "sentence": ["John", "works"],
            "starts": torch.tensor([1, 0]),
            "entity_spans": [(0, 1)],
            "entity_labels": ["PER"],
            "task_id": 1,
        }
    ]

    output = entity_collate_fn(batch)

    assert output["starts"].shape == (1, 2)
    assert output["spans"].shape == (1, 2, 2)
    assert output["spans"][0, 0, 1] == 1
    assert output["gold_labels"][0] == {(0, 1): "PER"}
    assert output["task_id"].shape == (1,)


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
        data = make_litbank(tag="split_0")
        batch = next(iter(data.train_loader))

        assert "sentences" in batch
        assert "span_loss_mask" in batch
        assert isinstance(batch["spans"], torch.Tensor)
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")


@patch("mentioned.data.load_dataset")
def test_make_litbank_entity(mock_load_dataset):

    # -----------------------------
    # Fake HF split
    # -----------------------------
    class FakeSplit(list):
        @property
        def column_names(self):
            return list(self[0].keys()) if self else []

    # -----------------------------
    # Fake HF dataset dict
    # -----------------------------
    class DummyDataset(dict):
        def map(self, fn, batched=False, remove_columns=None):
            mapped = {}

            for split_name, split_data in self.items():
                if not split_data:
                    mapped[split_name] = FakeSplit([])
                    continue

                if batched:
                    batch = {
                        "entities": [item["entities"] for item in split_data]
                    }

                    result = fn(batch)

                    new_split = []
                    for i in range(len(result["sentence"])):
                        new_split.append({
                            "sentence": result["sentence"][i],
                            "entity_spans": result["entity_spans"][i],
                            "entity_labels": result["entity_labels"][i],
                        })

                    mapped[split_name] = FakeSplit(new_split)
                else:
                    mapped[split_name] = FakeSplit(split_data)

            return DummyDataset(mapped)

    # -----------------------------
    # Fake data
    # -----------------------------
    fake_data = DummyDataset({
        "train": FakeSplit([
            {
                "entities": [
                    [
                        {"token": "John", "bio_tags": ["B-PER"]},
                        {"token": "Smith", "bio_tags": ["I-PER"]},
                    ]
                ]
            }
        ]),
        "validation": FakeSplit([]),
        "test": FakeSplit([]),
    })

    mock_load_dataset.return_value = fake_data

    # -----------------------------
    # Run function
    # -----------------------------
    data = make_litbank_entity()

    batch = next(iter(data.train_loader))
    print(batch)
    # -----------------------------
    # Assertions
    # -----------------------------
    assert "starts" in batch
    assert "spans" in batch
    assert "gold_labels" in batch

    # ensure entity span is present
    assert batch["spans"].sum() > 0
    assert batch["gold_labels"][0] == {(0, 1): "PER"}
