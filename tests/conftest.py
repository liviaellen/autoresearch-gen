"""Shared fixtures for autoresearch-gen tests."""

import os
import sys
from pathlib import Path

import pytest

# Add project root to path so we can import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def sample_program_md():
    return FIXTURES_DIR / "sample_program.md"


@pytest.fixture
def sample_train_py():
    return FIXTURES_DIR / "sample_train.py"


@pytest.fixture
def sample_experiments_tsv():
    return FIXTURES_DIR / "sample_experiments.tsv"


@pytest.fixture
def sample_results_tsv():
    return FIXTURES_DIR / "sample_results.tsv"


@pytest.fixture
def tmp_experiment(tmp_path):
    """Create a temporary experiment directory with sample files."""
    exp_dir = tmp_path / "test-experiment"
    exp_dir.mkdir()

    # Copy fixtures into the experiment dir
    for name in ("sample_experiments.tsv", "sample_results.tsv", "sample_program.md", "sample_train.py"):
        src = FIXTURES_DIR / name
        # Map fixture names to expected filenames
        dest_name = name.replace("sample_", "").replace("experiments_", "experiments.")
        if name == "sample_experiments.tsv":
            dest_name = "experiments.tsv"
        elif name == "sample_results.tsv":
            dest_name = "results.tsv"
        elif name == "sample_program.md":
            dest_name = "program.md"
        elif name == "sample_train.py":
            dest_name = "train.py"
        (exp_dir / dest_name).write_text(src.read_text())

    # Create pyproject.toml
    (exp_dir / "pyproject.toml").write_text(
        '[project]\nname = "test"\nversion = "0.1.0"\nrequires-python = ">=3.10"\n'
        'dependencies = ["mlx>=0.5", "numpy"]\n'
    )

    return exp_dir
