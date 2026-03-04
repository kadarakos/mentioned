"""Dataset preparation code."""

import torch

from typing import Callable
from collections import defaultdict
from dataclasses import dataclass

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class DataRegistry:
    _registry: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str) -> None:
        def decorator(func):
            cls._registry[name] = func
            return func
        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        return cls._registry[name]


@dataclass
class DataBlob:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    label2id: dict[int, str] | None = None


def build_label_mapping(loader: DataLoader) -> dict[str, int]:
    idx = 0
    label_to_id = {"O": idx}
    for batch in loader:
        for item in batch:
            labels = batch["gold_labels"]
            for annotation in labels:
                if annotation:
                    label = list(annotation.values())[0]
                    if label not in label_to_id:
                        idx += 1
                        label_to_id[label] = idx
    return label_to_id


class LitBankEntityDataset(Dataset):
    def __init__(self, hf_dataset: Dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        tokens = item["sentence"]
        spans = item["entity_spans"] or []

        # Create binary start mask for the 1D detector
        starts = torch.zeros(len(tokens), dtype=torch.long)
        for s, _ in spans:
            if s < len(tokens):
                starts[s] = 1

        return {
            "sentence": tokens,
            "starts": starts,
            "entity_spans": spans,
            "entity_labels": item["entity_labels"] or [],
            "task_id": 1
        }


class LitBankMentionDataset(Dataset):
    def __init__(self, hf_dataset: Dataset):
        self.dataset = hf_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        tokens = item["sentence"]
        # The ArrowDataset gives None for [].
        mentions = item["mentions"] if item["mentions"] is not None else []

        n_tokens = len(tokens)
        starts = torch.zeros(n_tokens, dtype=torch.long)
        span_labels = torch.zeros((n_tokens, n_tokens), dtype=torch.long)

        for s, e in mentions:
            # Ensure indices are within bounds (LitBank e is often inclusive)
            if s < n_tokens and e < n_tokens:
                starts[s] = 1
                span_labels[s, e] = 1

        return {
            "tokens": tokens,
            "starts": starts,
            "span_labels": span_labels,
            "task_id": 0,
        }


def mentions_by_sentence(example: dict) -> dict:
    mentions_per_sentence = defaultdict(list)
    for cluster in example["coref_chains"]:
        for mention in cluster:
            sent_idx, start, end = mention
            # In the ArrowDataset have to use str or byte as key.
            mentions_per_sentence[str(sent_idx)].append((start, end))
    example["mentions"] = mentions_per_sentence
    return example


def flatten_to_sentences(batch: dict) -> dict:
    new_batch = {"sentence": [], "mentions": []}

    # Ensure we are iterating over the lists in the batch
    for sentences, mentions_dict in zip(batch["sentences"], batch["mentions"]):
        # Some versions of datasets might save dicts as None if empty
        if mentions_dict is None:
            mentions_dict = {}

        for i, sent in enumerate(sentences):
            # Safe access: get the list of mentions or empty list
            sent_mentions = mentions_dict.get(str(i), [])

            new_batch["sentence"].append(sent)
            new_batch["mentions"].append(sent_mentions)

    return new_batch


def extract_spans_from_bio(sentence_tokens: list[dict]) -> tuple[list[tuple[int, int]], list[str]]:
    spans = []
    labels = []
    current_span = None

    for i, token_data in enumerate(sentence_tokens):
        tag = token_data["bio_tags"][0] if token_data["bio_tags"] else "O"

        if tag.startswith("B-"):
            if current_span:
                spans.append(tuple(current_span))
            label = tag.split("-")[1]
            current_span = [i, i]  # inclusive start/end
            labels.append(label)

        elif tag.startswith("I-") and current_span:
            current_span[1] = i  # inclusive extension

        else:
            if current_span:
                spans.append(tuple(current_span))
                current_span = None

    if current_span:
        spans.append(tuple(current_span))

    return spans, labels


def flatten_entities(batch: dict) -> dict:
    new_batch = {
        "sentence": [],
        "entity_spans": [],
        "entity_labels": []
    }
    for doc_sentences in batch["entities"]:
        for sentence_tokens in doc_sentences:
            tokens = [t["token"] for t in sentence_tokens]
            spans, labels = extract_spans_from_bio(sentence_tokens)

            new_batch["sentence"].append(tokens)
            new_batch["entity_spans"].append(spans)
            new_batch["entity_labels"].append(labels)
    return new_batch


def collate_fn(batch: list[dict]) -> dict:
    sentences = [item["tokens"] for item in batch]
    # Padding up to longest sentence.
    max_len = max(len(s) for s in sentences)
    starts_list = []  # 0 - 1 indicator for start tokens.
    spans_list = []  # 0 - 1 indicator for (start, end) pairs.

    for item in batch:
        curr_len = len(item["starts"])
        starts_list.append(item["starts"])
        padded_span = torch.zeros((max_len, max_len), dtype=torch.long)
        padded_span[:curr_len, :curr_len] = item["span_labels"]
        spans_list.append(padded_span)

    # 1D padding for token classification.
    starts_padded = pad_sequence(starts_list, batch_first=True, padding_value=-1)
    token_mask = starts_padded != -1
    starts_padded[starts_padded == -1] = 0

    # 2D padding for token-pair classification: B x N x N
    spans_padded = torch.stack(spans_list)
    # 2D length mask: B x N x 1 & B x 1 x N -> (B, N, N)
    valid_len_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
    # 2. Causal j >= i mask: B x N x N
    upper_tri_mask = torch.triu(
        torch.ones((max_len, max_len), dtype=torch.bool),
        diagonal=0,
    )
    # Mask all not start token positions: (B X N X 1)
    is_start_mask = starts_padded.unsqueeze(2).bool()
    # Full mask is "and"ing all masks together (like attention): B x N x N
    span_loss_mask = valid_len_mask & upper_tri_mask & is_start_mask

    return {
        "sentences": sentences,  # list[list[str]]
        "starts": starts_padded,  # (B, N) - Targets for start classifier
        "spans": spans_padded,  # (B, N, N) - Targets for span classifier
        "token_mask": token_mask,  # (B, N) - For 1D loss
        "span_loss_mask": span_loss_mask,  # (B, N, N) - For 2D loss
        "task_id": torch.tensor([item["task_id"] for item in batch]),
    }


def entity_collate_fn(batch: list[dict]) -> dict:
    # 1. Extract tokens using 'sentence' key
    sentences = [item["sentence"] for item in batch]
    max_len = max(len(s) for s in sentences)

    starts_list = []
    spans_list = []
    gold_label_maps = []

    for item in batch:
        starts_list.append(item["starts"])

        # 2. Build 2D binary matrix using 'entity_spans'
        binary_span_matrix = torch.zeros((max_len, max_len), dtype=torch.long)
        current_labels = {}

        # Use synchronized keys: entity_spans and entity_labels
        for (s, e), label_str in zip(item["entity_spans"], item["entity_labels"]):
            if s < max_len and e < max_len:
                binary_span_matrix[s, e] = 1
                current_labels[(s, e)] = label_str

        spans_list.append(binary_span_matrix)
        gold_label_maps.append(current_labels)

    # 3. Padding & Masking
    starts_padded = pad_sequence(starts_list, batch_first=True, padding_value=-1)
    token_mask = starts_padded != -1

    # Clean targets for loss (replace -1 with 0)
    starts_targets = starts_padded.clone()
    starts_targets[starts_targets == -1] = 0

    spans_padded = torch.stack(spans_list)

    valid_len_mask = token_mask.unsqueeze(2) & token_mask.unsqueeze(1)
    upper_tri_mask = torch.triu(torch.ones((max_len, max_len), dtype=torch.bool), 0)
    is_start_mask = starts_targets.unsqueeze(2).bool()
    span_loss_mask = valid_len_mask & upper_tri_mask & is_start_mask

    return {
        "sentences": sentences,
        "starts": starts_targets,
        "spans": spans_padded,
        "gold_labels": gold_label_maps,
        "token_mask": token_mask,
        "span_loss_mask": span_loss_mask,
        "task_id": torch.tensor([item["task_id"] for item in batch])
    }


def debug_print_entity_batch(batch: list[dict]) -> None:
    sentences = batch["sentences"]
    gold_labels_list = batch["gold_labels"]
    task_ids = batch["task_id"]

    print(f"--- Batch Debug (Size: {len(sentences)}) ---")

    for i, (tokens, labels_dict) in enumerate(zip(sentences, gold_labels_list)):
        task_name = "Entity" if task_ids[i] == 1 else "Mention"
        print(f"\n[Sentence {i}] Task: {task_name}")
        print(f"Text: {' '.join(tokens)}")

        if not labels_dict:
            print("  No entities found.")
            continue

        print("  Entities:")
        for (start, end), label in labels_dict.items():
            # Slice tokens: 'end' is exclusive in our logic
            entity_text = " ".join(tokens[start:end])
            print(f"    - [{label}] '{entity_text}' (indices: {start}:{end})")


@DataRegistry.register("litbank_mentions")
def make_litbank(
    repo_id: str = "coref-data/litbank_raw",
    tag: str = "split_0",
    batch_size: int = 4,
) -> DataBlob:
    """Reformat litbank to as a sentence-level mention-detection dataset."""
    litbank = load_dataset(repo_id, tag)
    litbank_sentences_mentions = litbank.map(mentions_by_sentence).map(
        flatten_to_sentences, batched=True, remove_columns=litbank["train"].column_names
    )
    no = 0
    for i in range(len(litbank_sentences_mentions["train"])):
        mentions = litbank_sentences_mentions["train"][i]["mentions"]
        # Check if None or empty
        if mentions is None or len(mentions) == 0:
            no += 1
    print(f"Training sentences without mentions: {no}.")
    bs = batch_size
    train = LitBankMentionDataset(litbank_sentences_mentions["train"])
    val = LitBankMentionDataset(litbank_sentences_mentions["validation"])
    test = LitBankMentionDataset(litbank_sentences_mentions["test"])
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    # Sanity check
    try:
        next(iter(train_loader))
    except Exception as e:
        raise e
    return DataBlob(train_loader, val_loader, test_loader)


@DataRegistry.register("litbank_entities")
def make_litbank_entity(
    repo_id: str = "coref-data/litbank_raw",
    tag: str = "split_0",
    batch_size: int = 4,
) -> DataBlob:
    litbank = load_dataset(repo_id, tag)
    entities_data = litbank.map(
        flatten_entities,
        batched=True,
        remove_columns=litbank["train"].column_names
    )
    bs = batch_size
    train = LitBankEntityDataset(entities_data["train"])
    val = LitBankEntityDataset(entities_data["validation"])
    test = LitBankEntityDataset(entities_data["test"])
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, collate_fn=entity_collate_fn)
    val_loader = DataLoader(val, batch_size=bs, shuffle=False, collate_fn=entity_collate_fn)
    test_loader = DataLoader(test, batch_size=bs, shuffle=False, collate_fn=entity_collate_fn)
    try:
        next(iter(train_loader))
    except Exception as e:
        raise e
    label2id = build_label_mapping(train_loader)
    return DataBlob(train_loader, val_loader, test_loader, label2id)
