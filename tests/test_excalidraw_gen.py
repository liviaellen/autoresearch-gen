"""Tests for excalidraw_gen.py — diagram generation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from excalidraw_gen import parse_train_py, parse_results_tsv, generate_diagram


# ---------------------------------------------------------------------------
# parse_train_py
# ---------------------------------------------------------------------------

class TestParseTrainPy:
    def test_parses_layers(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["n_layers"] == "8"

    def test_parses_d_model(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["d_model"] == "512"

    def test_parses_heads(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["n_heads"] == "8"

    def test_parses_kv_heads(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["n_kv_heads"] == "4"

    def test_parses_vocab_size(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["vocab_size"] == "32768"

    def test_parses_seq_len(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["seq_len"] == "1024"

    def test_parses_batch_size(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["batch_size"] == "16"

    def test_parses_learning_rate(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["learning_rate"] == "3e-4"

    def test_detects_gqa(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["attention"] == "GQA"

    def test_detects_rmsnorm(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["norm"] == "RMSNorm"

    def test_detects_swiglu(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["ffn"] == "SwiGLU"

    def test_detects_mlx_backend(self, sample_train_py):
        config = parse_train_py(str(sample_train_py))
        assert config["backend"] == "MLX"

    def test_missing_file(self):
        config = parse_train_py("/nonexistent/train.py")
        assert config == {}


# ---------------------------------------------------------------------------
# parse_results_tsv
# ---------------------------------------------------------------------------

class TestParseResultsTsv:
    def test_parses_rows(self, sample_results_tsv):
        rows = parse_results_tsv(str(sample_results_tsv))
        assert len(rows) == 3

    def test_row_has_keys(self, sample_results_tsv):
        rows = parse_results_tsv(str(sample_results_tsv))
        assert "commit" in rows[0]
        assert "val_bpb" in rows[0]
        assert "status" in rows[0]

    def test_baseline_first(self, sample_results_tsv):
        rows = parse_results_tsv(str(sample_results_tsv))
        assert rows[0]["status"] == "baseline"

    def test_missing_file(self):
        rows = parse_results_tsv("/nonexistent/results.tsv")
        assert rows == []


# ---------------------------------------------------------------------------
# generate_diagram
# ---------------------------------------------------------------------------

class TestGenerateDiagram:
    def test_returns_valid_excalidraw(self, tmp_experiment):
        doc = generate_diagram(str(tmp_experiment))
        assert doc["type"] == "excalidraw"
        assert doc["version"] == 2
        assert "elements" in doc
        assert len(doc["elements"]) > 0

    def test_has_title(self, tmp_experiment):
        doc = generate_diagram(str(tmp_experiment))
        texts = [e for e in doc["elements"] if e["type"] == "text"]
        text_contents = [e["text"] for e in texts]
        # Should contain the experiment name
        assert any("test-experiment" in t for t in text_contents)

    def test_has_architecture_box(self, tmp_experiment):
        doc = generate_diagram(str(tmp_experiment))
        ids = [e["id"] for e in doc["elements"]]
        assert "arch_box" in ids

    def test_has_results_box(self, tmp_experiment):
        doc = generate_diagram(str(tmp_experiment))
        ids = [e["id"] for e in doc["elements"]]
        assert "results_box" in ids

    def test_has_pipeline_steps(self, tmp_experiment):
        doc = generate_diagram(str(tmp_experiment))
        ids = [e["id"] for e in doc["elements"]]
        assert "pipe_prepare" in ids
        assert "pipe_train" in ids
        assert "pipe_eval" in ids

    def test_serializable_to_json(self, tmp_experiment):
        doc = generate_diagram(str(tmp_experiment))
        # Should not raise
        json_str = json.dumps(doc, indent=2)
        assert len(json_str) > 100

    def test_has_arrows(self, tmp_experiment):
        doc = generate_diagram(str(tmp_experiment))
        arrows = [e for e in doc["elements"] if e["type"] == "arrow"]
        assert len(arrows) >= 3  # at least arch->pipe, train->pipe, pipe->results
