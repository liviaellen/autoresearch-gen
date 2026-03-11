#!/usr/bin/env python3
"""
autoresearch-gen — Scaffold generator for autonomous pretraining research.

Generates a complete experiment directory (like Karpathy's autoresearch):
  - prepare.py   — data download, tokenizer, dataloader, eval (READ-ONLY)
  - train.py     — model + training loop (agent modifies this)
  - program.md   — autonomous experiment protocol

Supports PyTorch (CUDA) and MLX (Apple Silicon) backends.
Allows switching the LLM agent by passing --model and --api-key.

Usage:
    # Interactive — walks you through setup
    python gen.py

    # One-shot with flags
    python gen.py --output-dir experiments/mar10 --backend mlx --model claude-sonnet-4-20250514
"""

import argparse
import os
import platform
import sys

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

PREPARE_PT = '''\
"""
One-time data preparation for autoresearch experiments. (PyTorch / CUDA)
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = {time_budget}        # training time budget in seconds
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{{VAL_SHARD:05d}}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""\'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{{L}}\\p{{N}}]?+\\p{{L}}+|\\p{{N}}{{1,2}}| ?[^\\s\\p{{L}}\\p{{N}}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"""

SPECIAL_TOKENS = [f"<|reserved_{{i}}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    filename = f"shard_{{index:05d}}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True
    url = f"{{BASE_URL}}/{{filename}}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {{filename}}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {{attempt}}/{{max_attempts}} failed for {{filename}}: {{e}}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{{i:05d}}.parquet")))
    if existing == len(ids):
        print(f"Data: all {{len(ids)}} shards already downloaded at {{DATA_DIR}}")
        return
    needed = len(ids) - existing
    print(f"Data: downloading {{needed}} shards ({{existing}} already exist)...")
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)
    ok = sum(1 for r in results if r)
    print(f"Data: {{ok}}/{{len(ids)}} shards ready at {{DATA_DIR}}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {{TOKENIZER_DIR}}")
        return
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards. Download more data first.")
        sys.exit(1)
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {{bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {{name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}}
    enc = tiktoken.Encoding(name="rustbpe", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    t1 = time.time()
    print(f"Tokenizer: trained in {{t1 - t0:.1f}}s, saved to {{tokenizer_pkl}}")
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Tokenizer: saved token_bytes to {{token_bytes_path}}")
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {{test!r}} -> {{decoded!r}}"
    print(f"Tokenizer: sanity check passed (vocab_size={{enc.n_vocab}})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {{type(text)}}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[: B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T :].view(B, T)
    inputs = gpu_buffer[: B * T].view(B, T)
    targets = gpu_buffer[B * T :].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer")
    parser.add_argument("--num-shards", type=int, default=10, help="Training shards to download (-1 = all)")
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()
    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards
    print(f"Cache directory: {{CACHE_DIR}}\\n")
    download_data(num_shards, download_workers=args.download_workers)
    print()
    train_tokenizer()
    print("\\nDone! Ready to train.")
'''


PREPARE_MLX = '''\
"""
One-time data preparation for autoresearch experiments. (MLX / Apple Silicon)
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import argparse
import math
import os
import pickle
import sys
import time
from multiprocessing import Pool

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
TIME_BUDGET = {time_budget}
EVAL_TOKENS = 3 * 524288  # smaller for Apple Silicon

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542
VAL_SHARD = MAX_SHARD
VAL_FILENAME = f"shard_{{VAL_SHARD:05d}}.parquet"
VOCAB_SIZE = 8192

SPLIT_PATTERN = r"""\'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{{L}}\\p{{N}}]?+\\p{{L}}+|\\p{{N}}{{1,2}}| ?[^\\s\\p{{L}}\\p{{N}}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"""

SPECIAL_TOKENS = [f"<|reserved_{{i}}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    filename = f"shard_{{index:05d}}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True
    url = f"{{BASE_URL}}/{{filename}}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {{filename}}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {{attempt}}/{{max_attempts}} failed for {{filename}}: {{e}}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{{i:05d}}.parquet")))
    if existing == len(ids):
        print(f"Data: all {{len(ids)}} shards already downloaded at {{DATA_DIR}}")
        return
    needed = len(ids) - existing
    print(f"Data: downloading {{needed}} shards ({{existing}} already exist)...")
    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)
    ok = sum(1 for r in results if r)
    print(f"Data: {{ok}}/{{len(ids)}} shards ready at {{DATA_DIR}}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def train_tokenizer():
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    if os.path.exists(tokenizer_pkl) and os.path.exists(token_bytes_path):
        print(f"Tokenizer: already trained at {{TOKENIZER_DIR}}")
        return
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    parquet_files = list_parquet_files()
    if len(parquet_files) < 2:
        print("Tokenizer: need at least 2 data shards. Download more data first.")
        sys.exit(1)
    print("Tokenizer: training BPE tokenizer...")
    t0 = time.time()
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)
    pattern = tokenizer.get_pattern()
    mergeable_ranks = {{bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {{name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}}
    enc = tiktoken.Encoding(name="rustbpe", pat_str=pattern, mergeable_ranks=mergeable_ranks, special_tokens=special_tokens)
    with open(tokenizer_pkl, "wb") as f:
        pickle.dump(enc, f)
    t1 = time.time()
    print(f"Tokenizer: trained in {{t1 - t0:.1f}}s, saved to {{tokenizer_pkl}}")
    print("Tokenizer: building token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(token_str.encode("utf-8")))
    token_bytes_arr = np.array(token_bytes_list, dtype=np.int32)
    np.save(token_bytes_path, token_bytes_arr)
    print(f"Tokenizer: saved token_bytes to {{token_bytes_path}}")
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {{test!r}} -> {{decoded!r}}"
    print(f"Tokenizer: sanity check passed (vocab_size={{enc.n_vocab}})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {{type(text)}}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes():
    path = os.path.join(TOKENIZER_DIR, "token_bytes.npy")
    return mx.array(np.load(path))


def _document_batches(split, tokenizer_batch_size=128):
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i : i + tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    row_buffer = np.empty((B, row_capacity), dtype=np.int32)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()
                remaining = row_capacity - pos
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len
                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos : pos + len(doc)] = doc
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos : pos + remaining] = doc[:remaining]
                    pos += remaining
        inputs = mx.array(row_buffer[:, :-1])
        targets = mx.array(row_buffer[:, 1:])
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes()
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        logits = model(x)
        B, T, V = logits.shape
        loss_flat = mx.reshape(
            -mx.sum(mx.softmax(logits, axis=-1) * 0, axis=-1),  # placeholder
            (-1,)
        )
        # Cross-entropy per token
        logits_flat = mx.reshape(logits, (B * T, V))
        targets_flat = mx.reshape(y, (B * T,))
        log_probs = logits_flat - mx.logsumexp(logits_flat, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(log_probs, mx.expand_dims(targets_flat, -1), axis=-1).squeeze(-1)
        nbytes = token_bytes[targets_flat]
        mask = nbytes > 0
        total_nats += mx.sum(nll * mask).item()
        total_bytes += mx.sum(nbytes).item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer (MLX)")
    parser.add_argument("--num-shards", type=int, default=10, help="Training shards to download (-1 = all)")
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()
    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards
    print(f"Cache directory: {{CACHE_DIR}}\\n")
    download_data(num_shards, download_workers=args.download_workers)
    print()
    train_tokenizer()
    print("\\nDone! Ready to train.")
'''


TRAIN_PT = '''\
"""
Autoresearch pretraining script. Single-GPU, single-file. (PyTorch / CUDA)
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({{
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        }})
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos().bfloat16()[None, :, None, :]
        sin = freqs.sin().bfloat16()[None, :, None, :]
        return cos, sin

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)
        logits = self.lm_head(x).float()
        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=reduction,
            )
        return logits

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

DEPTH = {depth}
ASPECT_RATIO = 64
HEAD_DIM = 128
TOTAL_BATCH_SIZE = 2**19
LR = 0.001
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.3
DEVICE_BATCH_SIZE = {batch_size_pt}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
device = torch.device("cuda")
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()

base_dim = DEPTH * ASPECT_RATIO
model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
num_heads = model_dim // HEAD_DIM

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
    n_layer=DEPTH, n_head=num_heads, n_embd=model_dim,
)
print(f"Config: {{asdict(config)}}")

model = GPT(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {{num_params / 1e6:.1f}}M")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
grad_accum_steps = max(1, TOTAL_BATCH_SIZE // tokens_per_fwdbwd)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
model = torch.compile(model, dynamic=False)
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

print(f"Time budget: {{TIME_BUDGET}}s | Grad accum: {{grad_accum_steps}}")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_mult(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        return max(0.0, (1.0 - progress) / WARMDOWN_RATIO)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_train_start = time.time()
total_training_time = 0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    for _ in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        with autocast_ctx:
            loss = model(x, y)
        (loss / grad_accum_steps).backward()

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr_mult = get_lr_mult(progress)
    for g in optimizer.param_groups:
        g["lr"] = LR * lr_mult

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt = time.time() - t0
    if step > 5:
        total_training_time += dt

    remaining = max(0, TIME_BUDGET - total_training_time)
    print(f"\\rstep {{step:04d}} ({{100*progress:.0f}}%) | loss {{loss.item():.4f}} | dt {{dt*1000:.0f}}ms | rem {{remaining:.0f}}s  ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.disable()

    step += 1
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_bpb:          {{val_bpb:.6f}}")
print(f"training_seconds: {{total_training_time:.1f}}")
print(f"total_seconds:    {{t_end - t_start:.1f}}")
print(f"peak_vram_mb:     {{peak_vram_mb:.1f}}")
print(f"num_steps:        {{step}}")
print(f"num_params_M:     {{num_params / 1e6:.1f}}")
print(f"depth:            {{DEPTH}}")
'''


TRAIN_MLX = '''\
"""
Autoresearch pretraining script. Single-device, single-file. (MLX / Apple Silicon)
Usage: uv run train.py
"""

import gc
import math
import os
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
import numpy as np

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, evaluate_bpb, make_dataloader

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768


def norm(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rope = nn.RoPE(self.head_dim)

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        q = self.c_q(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        if mask is not None:
            scores = scores + mask
        w = mx.softmax(scores, axis=-1)
        y = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def __call__(self, x):
        x = self.c_fc(x)
        x = mx.maximum(x, 0) ** 2  # ReluSquared
        return self.c_proj(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def __call__(self, x, mask=None):
        x = x + self.attn(norm(x), mask)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        # Causal mask
        mask = mx.full((T, T), float("-inf"))
        mask = mx.triu(mask, k=1)

        x = self.wte(idx)
        x = norm(x)
        for block in self.blocks:
            x = block(x, mask)
        x = norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            logits_flat = logits.reshape(-1, logits.shape[-1])
            targets_flat = targets.reshape(-1)
            return mx.mean(nn.losses.cross_entropy(logits_flat, targets_flat))
        return logits

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

DEPTH = {depth}
ASPECT_RATIO = 64
HEAD_DIM = 128
TOTAL_BATCH_SIZE = 2**17  # smaller for Apple Silicon
LR = 0.001
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.3
DEVICE_BATCH_SIZE = {batch_size_mlx}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
mx.random.seed(42)

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()

base_dim = DEPTH * ASPECT_RATIO
model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
num_heads = model_dim // HEAD_DIM

config = GPTConfig(
    sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
    n_layer=DEPTH, n_head=num_heads, n_embd=model_dim,
)
print(f"Config: sequence_len={{config.sequence_len}}, vocab_size={{config.vocab_size}}, "
      f"n_layer={{config.n_layer}}, n_head={{config.n_head}}, n_embd={{config.n_embd}}")

model = GPT(config)
num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
print(f"Parameters: {{num_params / 1e6:.1f}}M")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
grad_accum_steps = max(1, TOTAL_BATCH_SIZE // tokens_per_fwdbwd)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")

print(f"Time budget: {{TIME_BUDGET}}s | Grad accum: {{grad_accum_steps}}")

# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_mult(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        return max(0.0, (1.0 - progress) / WARMDOWN_RATIO)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

optimizer = mx.optimizers.AdamW(learning_rate=LR, weight_decay=WEIGHT_DECAY, betas=[0.9, 0.95])
loss_and_grad_fn = nn.value_and_grad(model, lambda m, x, y: m(x, y))

t_train_start = time.time()
total_training_time = 0
step = 0

while True:
    t0 = time.time()

    total_loss = 0.0
    for _ in range(grad_accum_steps):
        x, y, epoch = next(train_loader)
        loss, grads = loss_and_grad_fn(model, x, y)
        grads = tree_map(lambda g: g / grad_accum_steps, grads)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        total_loss += loss.item()

    avg_loss = total_loss / grad_accum_steps

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr_mult = get_lr_mult(progress)
    optimizer.learning_rate = LR * lr_mult

    dt = time.time() - t0
    if step > 5:
        total_training_time += dt

    remaining = max(0, TIME_BUDGET - total_training_time)
    print(f"\\rstep {{step:04d}} ({{100*progress:.0f}}%) | loss {{avg_loss:.4f}} | dt {{dt*1000:.0f}}ms | rem {{remaining:.0f}}s  ", end="", flush=True)

    step += 1
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# Final eval
val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
print("---")
print(f"val_bpb:          {{val_bpb:.6f}}")
print(f"training_seconds: {{total_training_time:.1f}}")
print(f"total_seconds:    {{t_end - t_start:.1f}}")
print(f"num_steps:        {{step}}")
print(f"num_params_M:     {{num_params / 1e6:.1f}}")
print(f"depth:            {{DEPTH}}")
'''


PROGRAM_MD = '''\
# autoresearch

This is an experiment to have the LLM do its own research.

## Project Context

{project_context}

## Data

{data_description}

## Research Goals

{research_goals}

## Preferences

{preferences}

## Config

- **Agent LLM:** {model}
- **Backend:** {backend_label}
- **Tag:** {tag}

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/{tag}` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/{tag}` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single {device_label}. The training script runs for a **fixed time budget of {time_budget_min} minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time — it's always {time_budget_min} minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
num_steps:        953
num_params_M:     50.3
depth:            8
```

You can extract the key metric from the log file:

```
grep "^val_bpb:" run.log
```

## Logging results

Maintain TWO tab-separated log files (use tabs, NOT commas — commas break in descriptions):

### `experiments.tsv` — full history (every iteration)

Log EVERY experiment here, including reverts and crashes. This is the complete record.

```
commit\\tval_bpb\\tmemory_gb\\tstatus\\tdescription
a1b2c3d\\t0.997900\\t44.0\\tbaseline\\tbaseline
b2c3d4e\\t0.993200\\t44.2\\tkeep\\tincrease LR to 0.04
c3d4e5f\\t1.005000\\t44.0\\trevert\\tswitch to GeLU activation
d4e5f6g\\t0.000000\\t0.0\\tcrash\\tdouble model width (OOM)
```

### `results.tsv` — curated keeps only

Only `baseline` and `keep` rows go here. This is what you build on.

```
commit\\tval_bpb\\tmemory_gb\\tstatus\\tdescription
a1b2c3d\\t0.997900\\t44.0\\tbaseline\\tbaseline
b2c3d4e\\t0.993200\\t44.2\\tkeep\\tincrease LR to 0.04
```

### Columns

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3) — use 0.0 for crashes
4. status: `baseline`, `keep`, `revert`, or `crash`
5. short text description of what this experiment tried

You can add extra columns (e.g. `num_params_M`, `training_seconds`, `depth`) — the dashboard auto-detects any numeric column as a metric. Keep the header row consistent across both files.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/{tag}`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in BOTH tsv files: always append to `experiments.tsv`, and also append to `results.tsv` if the status is `baseline` or `keep` (NOTE: do not commit the tsv files, leave them untracked by git)
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~{time_budget_min} minutes total (+ a few seconds for startup and eval overhead). If a run exceeds {timeout_min} minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~{time_budget_min} minutes then you can run approx {experiments_per_hour}/hour. The user then wakes up to experimental results, all completed by you while they slept!
'''


GITIGNORE = '''\
__pycache__/
*.pyc
.venv/
.env
run.log
*.log
results.tsv
*.png
.DS_Store
uv.lock
'''


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(output_dir, backend, tag, model_name, api_key, time_budget, depth,
             batch_size_pt, batch_size_mlx,
             project_context="", data_description="", research_goals="", preferences=""):
    os.makedirs(output_dir, exist_ok=True)

    backend_label = "PyTorch (CUDA)" if backend == "pt" else "MLX (Apple Silicon)"
    device_label = "GPU" if backend == "pt" else "device (Apple Silicon)"

    # prepare.py
    if backend == "pt":
        prepare_code = PREPARE_PT.format(time_budget=time_budget)
    else:
        prepare_code = PREPARE_MLX.format(time_budget=time_budget)

    with open(os.path.join(output_dir, "prepare.py"), "w") as f:
        f.write(prepare_code)

    # train.py
    if backend == "pt":
        train_code = TRAIN_PT.format(
            depth=depth, batch_size_pt=batch_size_pt,
        )
    else:
        train_code = TRAIN_MLX.format(
            depth=depth, batch_size_mlx=batch_size_mlx,
        )

    with open(os.path.join(output_dir, "train.py"), "w") as f:
        f.write(train_code)

    # program.md (Karpathy-style)
    time_budget_min = time_budget // 60
    experiments_per_hour = 60 // max(time_budget_min, 1)
    program_md = PROGRAM_MD.format(
        tag=tag,
        model=model_name,
        backend_label=backend_label,
        device_label=device_label,
        time_budget_min=time_budget_min,
        timeout_min=time_budget * 2 // 60,
        experiments_per_hour=experiments_per_hour,
        project_context=project_context or "Language model pretraining experiment.",
        data_description=data_description or "Using Karpathy's climbmix-400b-shuffle dataset (~6.5K parquet shards hosted on HuggingFace). Data is downloaded and cached locally at ~/.cache/autoresearch/.",
        research_goals=research_goals or "Minimize val_bpb (bits per byte) on the validation set. Explore architecture, optimizer, and hyperparameter changes.",
        preferences=preferences or "Start with the baseline as-is. Keep changes simple and incremental. Prefer deleting complexity over adding it.",
    )
    with open(os.path.join(output_dir, "program.md"), "w") as f:
        f.write(program_md)

    # pyproject.toml
    if backend == "pt":
        deps = '''    "torch>=2.0",
    "numpy",
    "pyarrow",
    "requests",
    "rustbpe",
    "tiktoken",'''
    else:
        deps = '''    "mlx>=0.5",
    "numpy",
    "pyarrow",
    "requests",
    "rustbpe",
    "tiktoken",'''

    with open(os.path.join(output_dir, "pyproject.toml"), "w") as f:
        f.write(f'''[project]
name = "autoresearch-{tag}"
version = "0.1.0"
description = "Autonomous pretraining experiment: {tag}"
requires-python = ">=3.10"
dependencies = [
{deps}
]
''')

    # .gitignore
    with open(os.path.join(output_dir, ".gitignore"), "w") as f:
        f.write(GITIGNORE)

    # .env (if api key provided)
    if api_key:
        with open(os.path.join(output_dir, ".env"), "w") as f:
            # Detect provider from model name
            if "claude" in model_name.lower() or "anthropic" in model_name.lower():
                f.write(f"ANTHROPIC_API_KEY={api_key}\n")
            elif "gpt" in model_name.lower() or "o1" in model_name.lower() or "o3" in model_name.lower():
                f.write(f"OPENAI_API_KEY={api_key}\n")
            else:
                f.write(f"API_KEY={api_key}\n")

    return {
        "prepare": os.path.join(output_dir, "prepare.py"),
        "train": os.path.join(output_dir, "train.py"),
        "program": os.path.join(output_dir, "program.md"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def detect_backend():
    """Auto-detect: mlx on Apple Silicon, pt otherwise."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "pt"


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

LLM_PRESETS = {
    "1": ("claude-sonnet-4-20250514", "Anthropic", "ANTHROPIC_API_KEY"),
    "2": ("claude-opus-4-20250514", "Anthropic", "ANTHROPIC_API_KEY"),
    "3": ("gpt-4o", "OpenAI", "OPENAI_API_KEY"),
    "4": ("o3", "OpenAI", "OPENAI_API_KEY"),
    "5": ("deepseek-r1", "DeepSeek", "DEEPSEEK_API_KEY"),
}


def ask(prompt, default=None):
    """Prompt user, single line, with optional default."""
    if default:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return raw if raw else default
    else:
        while True:
            raw = input(f"  {prompt}: ").strip()
            if raw:
                return raw
            print("    (required)")


def ask_long(prompt, hint=""):
    """Prompt user for multi-line free text. Empty line or 'done' to finish."""
    print(f"\n  {prompt}")
    if hint:
        print(f"  {hint}")
    print("  (Type your answer. Press Enter twice or type 'done' when finished.)\n")
    lines = []
    while True:
        try:
            line = input("  > ")
        except EOFError:
            break
        if line.strip().lower() == "done":
            break
        if line == "" and lines and lines[-1] == "":
            lines.pop()  # remove trailing blank
            break
        lines.append(line)
    return "\n".join(lines).strip()


def interactive_setup():
    """Walk the user through setting up an experiment — context first."""
    print()
    print("=" * 60)
    print("  autoresearch-gen — scaffold a new experiment")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Step 1: Tell me about your project (free text)
    # ---------------------------------------------------------------
    project_context = ask_long(
        "Tell me about your project.",
        "What are you building or researching? Any relevant background the agent should know.",
    )

    # ---------------------------------------------------------------
    # Step 2: What data are you working with?
    # ---------------------------------------------------------------
    data_description = ask_long(
        "What data are you working with?",
        "Where does it come from? How big is it? What format? (e.g. 'climbmix-400b from HuggingFace, ~6.5K parquet shards')",
    )

    # ---------------------------------------------------------------
    # Step 3: What should the agent focus on?
    # ---------------------------------------------------------------
    research_goals = ask_long(
        "What should the agent try to optimize or explore?",
        "e.g. 'minimize val_bpb', 'try different attention patterns', 'find the smallest model that works'",
    )

    # ---------------------------------------------------------------
    # Step 4: Preferences for the starter code
    # ---------------------------------------------------------------
    preferences = ask_long(
        "Any preferences for how the scaffold should start?",
        "e.g. 'start small and simple', 'use a 4-layer model first', 'prefer attention-free architectures', 'keep it vanilla'",
    )

    # ---------------------------------------------------------------
    # Step 5: Tag and output dir
    # ---------------------------------------------------------------
    print()
    tag = ask("Experiment tag (e.g. mar10, small-gpt, ablation-lr)")
    default_dir = f"experiments/{tag}"
    output_dir = ask("Output directory", default=default_dir)

    # ---------------------------------------------------------------
    # Step 6: Backend
    # ---------------------------------------------------------------
    detected = detect_backend()
    detected_label = "MLX (Apple Silicon)" if detected == "mlx" else "PyTorch (CUDA)"
    print(f"\n  Detected: {detected_label}")
    backend = ask("Backend — pt or mlx", default=detected).lower().strip()
    if backend not in ("pt", "mlx"):
        print(f"  Unknown '{backend}', using '{detected}'")
        backend = detected

    # ---------------------------------------------------------------
    # Step 7: LLM model for the agent
    # ---------------------------------------------------------------
    print()
    print("  Which LLM should run the experiments?")
    print("    1) claude-sonnet-4-20250514  (Anthropic)")
    print("    2) claude-opus-4-20250514    (Anthropic)")
    print("    3) gpt-4o                    (OpenAI)")
    print("    4) o3                        (OpenAI)")
    print("    5) deepseek-r1               (DeepSeek)")
    print("    6) custom")
    print()
    model_choice = ask("Pick a number or enter model ID", default="1")

    if model_choice in LLM_PRESETS:
        model_name, provider, env_var = LLM_PRESETS[model_choice]
    elif model_choice == "6":
        model_name = ask("Model ID (e.g. mistral-large, gemini-2.0-flash)")
        provider = "custom"
        env_var = "API_KEY"
    else:
        model_name = model_choice
        provider = "custom"
        env_var = "API_KEY"

    print(f"  → {model_name}")

    # ---------------------------------------------------------------
    # Step 8: API key
    # ---------------------------------------------------------------
    existing_key = os.environ.get(env_var, "")
    if existing_key:
        print(f"  Found {env_var} in environment.")
        api_key = None
    else:
        print()
        key_input = ask(f"API key for {model_name} (Enter to skip)", default="")
        api_key = key_input if key_input else None
        if not api_key:
            print(f"  → Set {env_var} in env or .env later.")

    # ---------------------------------------------------------------
    # Step 9: Time budget and depth
    # ---------------------------------------------------------------
    print()
    budget_str = ask("Time budget per experiment (minutes)", default="5")
    try:
        time_budget = int(budget_str) * 60
    except ValueError:
        time_budget = 300

    depth_str = ask("Transformer depth (layers)", default="8")
    try:
        depth = int(depth_str)
    except ValueError:
        depth = 8

    batch_size_pt = 64
    batch_size_mlx = 16

    return {
        "output_dir": output_dir,
        "backend": backend,
        "tag": tag,
        "model_name": model_name,
        "api_key": api_key,
        "time_budget": time_budget,
        "depth": depth,
        "batch_size_pt": batch_size_pt,
        "batch_size_mlx": batch_size_mlx,
        "project_context": project_context,
        "data_description": data_description,
        "research_goals": research_goals,
        "preferences": preferences,
    }


def print_summary(output_dir, backend, model_name, time_budget, depth):
    backend_label = "PyTorch (CUDA)" if backend == "pt" else "MLX (Apple Silicon)"
    print()
    print("=" * 60)
    print(f"  Scaffolded in {output_dir}/")
    print("=" * 60)
    print()
    print(f"  Backend:     {backend_label}")
    print(f"  Agent LLM:   {model_name}")
    print(f"  Time budget: {time_budget}s ({time_budget // 60} min)")
    print(f"  Depth:       {depth} layers")
    print()
    print(f"  prepare.py   ← data + tokenizer + eval (DO NOT MODIFY)")
    print(f"  train.py     ← model + training loop (agent modifies)")
    print(f"  program.md   ← experiment protocol (context + rules + loop)")
    print()
    print(f"  Next steps:")
    print(f"    cd {output_dir}")
    print(f"    uv run prepare.py       # download data + train tokenizer")
    print(f"    uv run train.py         # run baseline")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="autoresearch-gen: scaffold autonomous pretraining experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Interactive — walks you through it, asks about your project
  python gen.py

  # One-shot with flags
  python gen.py --output-dir exp/mar10 --backend mlx
  python gen.py --output-dir exp/mar10 --model gpt-4o --api-key sk-...
  python gen.py --output-dir exp/mar10 --backend pt --depth 12 --time-budget 600 \\
      --context "training a small GPT on code data" \\
      --data "custom tokenized code corpus, 10M tokens" \\
      --goals "find the best LR schedule for small models" \\
      --prefs "start with 4 layers, keep it minimal"
""",
    )
    parser.add_argument("--output-dir", default=None, help="Directory to generate into")
    parser.add_argument("--backend", choices=["pt", "mlx"], default=None,
                        help="Training backend (default: auto-detect)")
    parser.add_argument("--tag", default=None, help="Experiment tag (default: dir name)")
    parser.add_argument("--model", default=None,
                        help="LLM for the agent (default: claude-sonnet-4-20250514)")
    parser.add_argument("--api-key", default=None,
                        help="API key (or set env var)")
    parser.add_argument("--time-budget", type=int, default=None,
                        help="Time budget in seconds (default: 300)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Transformer depth (default: 8)")
    parser.add_argument("--batch-size-pt", type=int, default=64)
    parser.add_argument("--batch-size-mlx", type=int, default=16)
    # Context flags for one-shot mode
    parser.add_argument("--context", default=None,
                        help="Project context — what you're building/researching")
    parser.add_argument("--data", default=None,
                        help="Data description — source, format, size")
    parser.add_argument("--goals", default=None,
                        help="Research goals — what to optimize or explore")
    parser.add_argument("--prefs", default=None,
                        help="Scaffold preferences — how starter code should look")

    args = parser.parse_args()

    # If --output-dir not given, go interactive
    if args.output_dir is None:
        config = interactive_setup()
    else:
        # One-shot mode
        config = {
            "output_dir": args.output_dir,
            "backend": args.backend or detect_backend(),
            "tag": args.tag or os.path.basename(os.path.normpath(args.output_dir)),
            "model_name": args.model or "claude-sonnet-4-20250514",
            "api_key": args.api_key,
            "time_budget": args.time_budget or 300,
            "depth": args.depth or 8,
            "batch_size_pt": args.batch_size_pt,
            "batch_size_mlx": args.batch_size_mlx,
            "project_context": args.context or "",
            "data_description": args.data or "",
            "research_goals": args.goals or "",
            "preferences": args.prefs or "",
        }

    files = generate(**config)

    print_summary(
        config["output_dir"], config["backend"], config["model_name"],
        config["time_budget"], config["depth"],
    )


if __name__ == "__main__":
    main()
