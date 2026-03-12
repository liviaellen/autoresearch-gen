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
    print(f"\rstep {{step:04d}} ({{100*progress:.0f}}%) | loss {{loss.item():.4f}} | dt {{dt*1000:.0f}}ms | rem {{remaining:.0f}}s  ", end="", flush=True)

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
