"""Sample train.py for testing parse_train_py."""
import mlx.core as mx
import mlx.nn as nn

MAX_SEQ_LEN = 1024
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
TIME_BUDGET = 300

n_layers = 8
d_model = 512
n_heads = 8
n_kv_heads = 4
vocab_size = 32768

# Uses GQA attention
# RMSNorm for normalization
# SwiGLU FFN activation
