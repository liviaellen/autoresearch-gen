"""Tests for dashboard.py — utility functions (no Streamlit required)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard import metric_direction, is_metric_col, parse_program_md, load_tsv


# ---------------------------------------------------------------------------
# metric_direction
# ---------------------------------------------------------------------------

class TestMetricDirection:
    def test_loss_is_lower(self):
        assert metric_direction("val_loss") == "lower"

    def test_bpb_is_lower(self):
        assert metric_direction("val_bpb") == "lower"

    def test_rmse_is_lower(self):
        assert metric_direction("cv_rmse") == "lower"

    def test_mse_is_lower(self):
        assert metric_direction("mse") == "lower"

    def test_mae_is_lower(self):
        assert metric_direction("mae") == "lower"

    def test_perplexity_is_lower(self):
        assert metric_direction("perplexity") == "lower"

    def test_accuracy_is_higher(self):
        assert metric_direction("accuracy") == "higher"

    def test_f1_is_higher(self):
        assert metric_direction("f1") == "higher"

    def test_auc_is_higher(self):
        assert metric_direction("auc") == "higher"

    def test_precision_is_higher(self):
        assert metric_direction("precision") == "higher"

    def test_recall_is_higher(self):
        assert metric_direction("recall") == "higher"

    def test_r2_is_higher(self):
        assert metric_direction("r2") == "higher"

    def test_unknown_defaults_lower(self):
        assert metric_direction("some_metric") == "lower"

    def test_case_insensitive(self):
        assert metric_direction("Val_BPB") == "lower"
        assert metric_direction("ACCURACY") == "higher"


# ---------------------------------------------------------------------------
# is_metric_col
# ---------------------------------------------------------------------------

class TestIsMetricCol:
    def test_numeric_col_is_metric(self):
        df = pd.DataFrame({"val_bpb": [1.0, 2.0], "status": ["keep", "revert"]})
        assert is_metric_col("val_bpb", df) is True

    def test_status_not_metric(self):
        df = pd.DataFrame({"val_bpb": [1.0], "status": ["keep"]})
        assert is_metric_col("status", df) is False

    def test_commit_not_metric(self):
        df = pd.DataFrame({"commit": ["abc123"], "val_bpb": [1.0]})
        assert is_metric_col("commit", df) is False

    def test_description_not_metric(self):
        df = pd.DataFrame({"description": ["test"], "val_bpb": [1.0]})
        assert is_metric_col("description", df) is False

    def test_string_col_not_metric(self):
        df = pd.DataFrame({"name": ["a", "b"], "score": [1.0, 2.0]})
        assert is_metric_col("name", df) is False

    def test_memory_gb_is_metric(self):
        df = pd.DataFrame({"memory_gb": [27.4, 29.7]})
        assert is_metric_col("memory_gb", df) is True


# ---------------------------------------------------------------------------
# parse_program_md
# ---------------------------------------------------------------------------

class TestParseProgramMd:
    def test_parses_tag(self, sample_program_md):
        meta = parse_program_md(sample_program_md)
        assert meta["tag"] == "attention-free"

    def test_parses_backend(self, sample_program_md):
        meta = parse_program_md(sample_program_md)
        assert "MLX" in meta["backend"]

    def test_parses_agent_llm(self, sample_program_md):
        meta = parse_program_md(sample_program_md)
        assert "claude" in meta["agent_llm"].lower()

    def test_parses_context(self, sample_program_md):
        meta = parse_program_md(sample_program_md)
        assert "context" in meta
        assert len(meta["context"]) > 0

    def test_parses_goals(self, sample_program_md):
        meta = parse_program_md(sample_program_md)
        assert "goals" in meta
        assert "val_bpb" in meta["goals"].lower()

    def test_detects_val_bpb_primary(self, sample_program_md):
        meta = parse_program_md(sample_program_md)
        assert meta.get("primary_metric") == "val_bpb"

    def test_detects_direction(self, sample_program_md):
        meta = parse_program_md(sample_program_md)
        assert meta.get("direction") == "lower"

    def test_missing_file(self, tmp_path):
        meta = parse_program_md(tmp_path / "nonexistent.md")
        assert meta == {}


# ---------------------------------------------------------------------------
# load_tsv
# ---------------------------------------------------------------------------

class TestLoadTsv:
    def test_loads_experiments(self, sample_experiments_tsv):
        df = load_tsv(sample_experiments_tsv)
        assert df is not None
        assert len(df) == 5
        assert "val_bpb" in df.columns
        assert "status" in df.columns
        assert "commit" in df.columns

    def test_loads_results(self, sample_results_tsv):
        df = load_tsv(sample_results_tsv)
        assert df is not None
        assert len(df) == 3

    def test_replaces_dash_with_nan(self, sample_experiments_tsv):
        df = load_tsv(sample_experiments_tsv)
        # The crash row has "-" for memory_gb which should become NaN
        assert pd.isna(df.loc[df["status"] == "crash", "memory_gb"].iloc[0])

    def test_numeric_conversion(self, sample_experiments_tsv):
        df = load_tsv(sample_experiments_tsv)
        assert pd.api.types.is_numeric_dtype(df["val_bpb"])

    def test_missing_file(self, tmp_path):
        result = load_tsv(tmp_path / "nonexistent.tsv")
        assert result is None

    def test_nan_in_val_bpb(self, sample_experiments_tsv):
        df = load_tsv(sample_experiments_tsv)
        crash_row = df[df["status"] == "crash"]
        assert len(crash_row) == 1
        assert pd.isna(crash_row["val_bpb"].iloc[0])

    def test_baseline_val(self, sample_experiments_tsv):
        df = load_tsv(sample_experiments_tsv)
        baseline = df[df["status"] == "baseline"]
        assert len(baseline) == 1
        assert baseline["val_bpb"].iloc[0] == pytest.approx(2.109738)
