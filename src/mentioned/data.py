"""Dataset preparation code."""

import torch


from collections import defaultdict

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def mentions_by_sentence(example):
    mentions_per_sentence = defaultdict(list)
    for cluster in example["coref_chains"]:
        for mention in cluster:
            sent_idx, start, end = mention
            # In the ArrowDataset have to use str or byte as key.
            mentions_per_sentence[str(sent_idx)].append((start, end))
    example["mentions"] = mentions_per_sentence
    return example


def flatten_to_sentences(batch):
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


class LitBankMentionDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
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
        }


def collate_fn(batch):
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
    }


def make_litbank(
    repo_id: str = "coref-data/litbank_raw",
    tag: str = "split_0",
) -> tuple[DataLoader, DataLoader, DataLoader]:
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
    train = LitBankMentionDataset(litbank_sentences_mentions["train"])
    val = LitBankMentionDataset(litbank_sentences_mentions["validation"])
    test = LitBankMentionDataset(litbank_sentences_mentions["test"])
    train_loader = DataLoader(train, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test, batch_size=4, shuffle=False, collate_fn=collate_fn)
    # Sanity check
    try:
        next(iter(train_loader))
    except Exception as e:
        raise e
    return train_loader, val_loader, test_loader
