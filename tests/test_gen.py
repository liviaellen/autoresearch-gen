"""Tests for gen.py — scaffold generator."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gen import (
    _detect_provider,
    _strip_fences,
    _parse_train_output,
    detect_backend,
    generate,
    resolve_api_key,
)


# ---------------------------------------------------------------------------
# _detect_provider
# ---------------------------------------------------------------------------

class TestDetectProvider:
    def test_anthropic_claude(self):
        provider, env_var = _detect_provider("claude-sonnet-4-20250514")
        assert provider == "anthropic"
        assert env_var == "ANTHROPIC_API_KEY"

    def test_anthropic_keyword(self):
        provider, _ = _detect_provider("anthropic-model-v1")
        assert provider == "anthropic"

    def test_openai_gpt(self):
        provider, env_var = _detect_provider("gpt-4o")
        assert provider == "openai"
        assert env_var == "OPENAI_API_KEY"

    def test_openai_o3(self):
        provider, _ = _detect_provider("o3")
        assert provider == "openai"

    def test_deepseek(self):
        provider, env_var = _detect_provider("deepseek-r1")
        assert provider == "deepseek"
        assert env_var == "DEEPSEEK_API_KEY"

    def test_litellm_override(self):
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://localhost:4000"}):
            provider, env_var = _detect_provider("any-model")
            assert provider == "litellm"
            assert env_var == "LITELLM_API_KEY"

    def test_unknown_defaults_openai(self):
        provider, _ = _detect_provider("some-random-model")
        assert provider == "openai"


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_no_fences(self):
        assert _strip_fences("hello world") == "hello world"

    def test_python_fences(self):
        text = "```python\nprint('hi')\n```"
        assert _strip_fences(text) == "print('hi')"

    def test_plain_fences(self):
        text = "```\nsome code\n```"
        assert _strip_fences(text) == "some code"

    def test_multiline(self):
        text = "```python\nline1\nline2\nline3\n```"
        assert _strip_fences(text) == "line1\nline2\nline3"

    def test_no_trailing_fence(self):
        text = "```python\ncode here"
        assert _strip_fences(text) == "code here"


# ---------------------------------------------------------------------------
# _parse_train_output
# ---------------------------------------------------------------------------

class TestParseTrainOutput:
    def test_basic_parse(self):
        output = "some log line\n---\nval_bpb: 1.5432\ntraining_seconds: 120.5\n---\n"
        result = _parse_train_output(output)
        assert result["val_bpb"] == pytest.approx(1.5432)
        assert result["training_seconds"] == pytest.approx(120.5)

    def test_with_string_values(self):
        output = "---\nval_bpb: 1.23\ndevice: mps\n"
        result = _parse_train_output(output)
        assert result["val_bpb"] == pytest.approx(1.23)
        assert result["device"] == "mps"

    def test_empty_output(self):
        result = _parse_train_output("")
        assert result == {}

    def test_no_separator(self):
        result = _parse_train_output("just some text\nno separator here\n")
        assert result == {}

    def test_multiple_metrics(self):
        output = "---\nval_bpb: 1.08\nnum_params_M: 25.3\npeak_vram_mb: 4096\nnum_steps: 500\n"
        result = _parse_train_output(output)
        assert result["val_bpb"] == pytest.approx(1.08)
        assert result["num_params_M"] == pytest.approx(25.3)
        assert result["peak_vram_mb"] == pytest.approx(4096)
        assert result["num_steps"] == pytest.approx(500)


# ---------------------------------------------------------------------------
# detect_backend
# ---------------------------------------------------------------------------

class TestDetectBackend:
    def test_returns_valid_backend(self):
        result = detect_backend()
        assert result in ("pt", "mlx")


# ---------------------------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------------------------

class TestResolveApiKey:
    def test_cli_key_takes_priority(self):
        key = resolve_api_key("claude-sonnet-4-20250514", cli_key="sk-test-123", interactive=False)
        assert key == "sk-test-123"

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-env-key"}):
            key = resolve_api_key("claude-sonnet-4-20250514", interactive=False)
            assert key == "sk-env-key"

    def test_no_key_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing keys
            for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "LITELLM_API_BASE"):
                os.environ.pop(k, None)
            key = resolve_api_key("claude-sonnet-4-20250514", interactive=False)
            assert key is None

    def test_litellm_no_key_ok(self):
        with patch.dict(os.environ, {"LITELLM_API_BASE": "http://localhost:4000"}, clear=False):
            key = resolve_api_key("any-model", interactive=False)
            assert key is not None


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_creates_all_files(self, tmp_path):
        output_dir = str(tmp_path / "test-scaffold")
        generate(
            output_dir=output_dir,
            backend="mlx",
            tag="test-tag",
            model_name="claude-sonnet-4-20250514",
            api_key=None,
            time_budget=300,
            depth=8,
            batch_size_pt=64,
            batch_size_mlx=16,
            use_llm=False,
        )
        files = os.listdir(output_dir)
        assert "prepare.py" in files
        assert "train.py" in files
        assert "program.md" in files
        assert "pyproject.toml" in files
        assert ".gitignore" in files

    def test_train_py_contains_depth(self, tmp_path):
        output_dir = str(tmp_path / "test-depth")
        generate(
            output_dir=output_dir,
            backend="mlx",
            tag="depth-test",
            model_name="gpt-4o",
            api_key=None,
            time_budget=300,
            depth=12,
            batch_size_pt=64,
            batch_size_mlx=16,
            use_llm=False,
        )
        train_code = (tmp_path / "test-depth" / "train.py").read_text()
        assert "12" in train_code

    def test_program_md_contains_tag(self, tmp_path):
        output_dir = str(tmp_path / "test-tag")
        generate(
            output_dir=output_dir,
            backend="mlx",
            tag="my-experiment",
            model_name="gpt-4o",
            api_key=None,
            time_budget=300,
            depth=8,
            batch_size_pt=64,
            batch_size_mlx=16,
            use_llm=False,
        )
        program = (tmp_path / "test-tag" / "program.md").read_text()
        assert "my-experiment" in program

    def test_pyproject_has_correct_deps_mlx(self, tmp_path):
        output_dir = str(tmp_path / "test-mlx")
        generate(
            output_dir=output_dir,
            backend="mlx",
            tag="mlx-test",
            model_name="gpt-4o",
            api_key=None,
            time_budget=300,
            depth=8,
            batch_size_pt=64,
            batch_size_mlx=16,
            use_llm=False,
        )
        pyproject = (tmp_path / "test-mlx" / "pyproject.toml").read_text()
        assert "mlx" in pyproject

    def test_pyproject_has_correct_deps_pt(self, tmp_path):
        output_dir = str(tmp_path / "test-pt")
        generate(
            output_dir=output_dir,
            backend="pt",
            tag="pt-test",
            model_name="gpt-4o",
            api_key=None,
            time_budget=300,
            depth=8,
            batch_size_pt=64,
            batch_size_mlx=16,
            use_llm=False,
        )
        pyproject = (tmp_path / "test-pt" / "pyproject.toml").read_text()
        assert "torch" in pyproject

    def test_context_in_program_md(self, tmp_path):
        output_dir = str(tmp_path / "test-context")
        generate(
            output_dir=output_dir,
            backend="mlx",
            tag="ctx-test",
            model_name="gpt-4o",
            api_key=None,
            time_budget=300,
            depth=8,
            batch_size_pt=64,
            batch_size_mlx=16,
            project_context="Testing attention-free architectures",
            use_llm=False,
        )
        program = (tmp_path / "test-context" / "program.md").read_text()
        assert "attention-free" in program.lower()

    def test_git_repo_initialized(self, tmp_path):
        output_dir = str(tmp_path / "test-git")
        generate(
            output_dir=output_dir,
            backend="mlx",
            tag="git-test",
            model_name="gpt-4o",
            api_key=None,
            time_budget=300,
            depth=8,
            batch_size_pt=64,
            batch_size_mlx=16,
            use_llm=False,
        )
        assert (tmp_path / "test-git" / ".git").exists()
