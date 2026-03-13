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
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request


def _load_dotenv(path=None):
    """Load .env file into os.environ (stdlib only, no dependencies)."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            # Don't overwrite existing env vars
            if key and value and key not in os.environ:
                os.environ[key] = value


_load_dotenv()

# ---------------------------------------------------------------------------
# Templates (loaded from templates/ directory)
# ---------------------------------------------------------------------------

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def _load_template(name):
    """Load a template file from the templates/ directory."""
    with open(os.path.join(_TEMPLATE_DIR, name)) as f:
        return f.read()


PREPARE_PT = _load_template("prepare_pt.py")
PREPARE_MLX = _load_template("prepare_mlx.py")
TRAIN_PT = _load_template("train_pt.py")
TRAIN_MLX = _load_template("train_mlx.py")
PROGRAM_MD = _load_template("program.md")
GITIGNORE = _load_template("gitignore")


# ---------------------------------------------------------------------------
# LLM API
# ---------------------------------------------------------------------------

def _detect_provider(model_name):
    """Return ('anthropic'|'openai'|'deepseek'|'litellm', env_var_name) for a model."""
    # If LiteLLM proxy is configured, route everything through it
    if os.environ.get("LITELLM_API_BASE"):
        return "litellm", "LITELLM_API_KEY"
    m = model_name.lower()
    if "claude" in m or "anthropic" in m:
        return "anthropic", "ANTHROPIC_API_KEY"
    if "deepseek" in m:
        return "deepseek", "DEEPSEEK_API_KEY"
    # Default to OpenAI for gpt, o1, o3, etc.
    return "openai", "OPENAI_API_KEY"


def resolve_api_key(model_name, cli_key=None, interactive=True):
    """Resolve API key from CLI flag, then env var, then prompt. Returns key or None."""
    if cli_key:
        return cli_key
    provider, env_var = _detect_provider(model_name)
    env_key = os.environ.get(env_var)
    if env_key:
        return env_key
    # LiteLLM proxy may not need a key (e.g. local ollama)
    if provider == "litellm":
        return os.environ.get("LITELLM_API_KEY", "no-key-needed")
    if interactive and sys.stdin.isatty():
        print(f"\n  No API key found for {model_name} (checked --api-key flag and ${env_var}).")
        key_input = input(f"  Enter your {provider.capitalize()} API key (or press Enter to skip): ").strip()
        if key_input:
            return key_input
    return None


def call_llm(model_name, api_key, system_prompt, user_prompt):
    """Call an LLM and return the text response. Supports Anthropic & OpenAI-compatible APIs."""
    provider, _ = _detect_provider(model_name)

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": model_name,
            "max_tokens": 8192,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
    else:
        if provider == "litellm":
            base = os.environ["LITELLM_API_BASE"].rstrip("/")
            url = f"{base}/chat/completions"
        elif provider == "deepseek":
            url = "https://api.deepseek.com/v1/chat/completions"
        else:
            url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model_name,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    if provider == "anthropic":
        return result["content"][0]["text"]
    else:
        return result["choices"][0]["message"]["content"]


def _strip_fences(text):
    """Remove markdown code fences if the LLM wrapped its output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def gather_deep_context(model_name, api_key, project_context, data_description,
                        research_goals, preferences, backend, max_rounds=2):
    """Use the LLM to ask smart follow-up questions before scaffolding.

    Runs a multi-turn interview: the LLM sees what the user provided,
    identifies gaps or vague areas, and asks targeted follow-ups.
    Returns an enriched context dict.
    """
    backend_label = "MLX (Apple Silicon)" if backend == "mlx" else "PyTorch (CUDA)"

    system_prompt = """\
You are an expert ML research assistant helping set up a pretraining experiment.

The user has given you some initial context about their project. Your job is to
fill in the gaps with your best recommendations — architecture, model size, LR,
sequence length, anything that affects the generated code.

Write a SHORT summary (2-4 sentences) of what you'd build given their context.
Be concrete — name specific values (layers, embed dim, LR, schedule, etc.).
End with "What do you think?"

Example output:
I'd go with a GPT-style model — 8 layers, 512 embed dim, 8 heads, GQA with
sliding window attention. Cosine LR decay from 3e-4 to 3e-5 with linear warmup.
Sequence length 1024, should fit comfortably in 48GB. What do you think?

Rules:
- Do NOT repeat things already clearly answered.
- ALWAYS propose concrete values, never ask bare questions.
- Keep it brief — no bullet lists, no preamble, just the summary.
- If you already have enough info, say "No further questions."
"""

    user_prompt = f"""\
## What the user provided so far
- Project context: {project_context or "(empty)"}
- Data: {data_description or "(empty)"}
- Research goals: {research_goals or "(empty)"}
- Preferences: {preferences or "(empty)"}
- Backend: {backend_label}

What follow-up questions would help you design a better experiment?"""

    enriched = {
        "project_context": project_context or "",
        "data_description": data_description or "",
        "research_goals": research_goals or "",
        "preferences": preferences or "",
    }
    follow_ups = []

    for round_num in range(max_rounds):
        try:
            questions = call_llm(model_name, api_key, system_prompt, user_prompt)
        except Exception as e:
            print(f"  (Could not generate follow-ups: {e})")
            break

        # Check if LLM thinks we have enough info
        if "no further questions" in questions.lower() or "looks good" in questions.lower():
            print("  Got enough context.")
            break

        print(f"\n  Here's what I'm thinking:\n")
        for line in questions.strip().splitlines():
            if line.strip():
                print(f"  {line}")
        print()
        print("  Press Enter to accept, or type changes.\n")
        try:
            answers = input("  > ").strip()
        except EOFError:
            answers = ""

        if not answers:
            # User accepted suggestions — feed them back as confirmation
            answers = "Looks good, go with your suggestions."
        if answers.lower() in ("skip", "done"):
            break

        follow_ups.append(f"Q: {questions}\nA: {answers}")

        # Update the user prompt for next round with the new info
        all_follow_ups = "\n\n".join(follow_ups)
        user_prompt = f"""\
## What the user provided so far
- Project context: {enriched['project_context']}
- Data: {enriched['data_description']}
- Research goals: {enriched['research_goals']}
- Preferences: {enriched['preferences']}
- Backend: {backend_label}

## Follow-up clarifications
{all_follow_ups}

The user just responded to your suggestions. Do you have enough to build a good scaffold?
If yes, respond with "No further questions."
If something critical is still unclear, propose 1-2 more suggestions in the same format:
  1. [Topic] — I'm thinking [suggestion]. What do you think?"""

    # Append follow-up Q&A as a distinct section in preferences
    if follow_ups:
        follow_up_text = "\n\n".join(follow_ups)
        separator = "\n\n" if enriched["preferences"] else ""
        enriched["preferences"] += f"{separator}## Clarifications from interview\n{follow_up_text}"

    return enriched


def customize_with_llm(base_template, file_type, model_name, api_key,
                       project_context, data_description, research_goals,
                       preferences, backend, time_budget, depth):
    """Send the base template + user context to the LLM and return customized content."""
    backend_label = "PyTorch (CUDA)" if backend == "pt" else "MLX (Apple Silicon)"

    system_prompt = f"""\
You are a scaffold generator for autonomous ML pretraining experiments.
You will receive a base {file_type} template and the user's project context.
Return ONLY the customized file content — no markdown fences, no explanations, no preamble.
The output must be valid {"Python code" if file_type.endswith(".py") else "Markdown"} that works correctly.

Rules:
- Keep the core structure and imports intact.
- Adjust architecture, hyperparameters, comments, and documentation to match the user's goals.
- For train.py: you may change depth, LR, batch size, model dimensions, MLP ratio, optimizer, etc.
  Do NOT change the imports from prepare.py or the output format (val_bpb, training_seconds, etc.).
- For program.md: enrich context sections, tailor experiment suggestions to the user's goals.
- Keep it clean and production-ready. No placeholder text."""

    user_prompt = f"""\
## User Context
- Project: {project_context or "Not specified"}
- Data: {data_description or "Not specified"}
- Research goals: {research_goals or "Not specified"}
- Preferences: {preferences or "Not specified"}
- Backend: {backend_label}
- Time budget: {time_budget}s ({time_budget // 60} min)
- Depth: {depth} layers

## Base Template ({file_type})
{base_template}

Customize this template based on the user's context above. Return the full file content."""

    text = call_llm(model_name, api_key, system_prompt, user_prompt)
    return _strip_fences(text)


def fix_with_llm(broken_code, error_output, model_name, api_key, backend):
    """Send broken code + error back to the LLM and ask it to fix."""
    backend_label = "PyTorch (CUDA)" if backend == "pt" else "MLX (Apple Silicon)"

    system_prompt = """\
You are debugging a train.py file for an ML pretraining experiment.
The code crashed during baseline validation. You will receive the broken code
and the error output. Fix the issue and return the COMPLETE corrected file.

Rules:
- Return ONLY the fixed Python code — no markdown fences, no explanations.
- Do NOT change imports from prepare.py or the output format (val_bpb, etc.).
- Fix the actual bug — don't just add try/except to suppress it.
- Keep the same architecture intent, just make it work."""

    user_prompt = f"""\
## Backend
{backend_label}

## Broken train.py
{broken_code}

## Error Output (last 50 lines)
{error_output}

Fix the code and return the complete corrected train.py."""

    text = call_llm(model_name, api_key, system_prompt, user_prompt)
    return _strip_fences(text)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate(output_dir, backend, tag, model_name, api_key, time_budget, depth,
             batch_size_pt, batch_size_mlx,
             project_context="", data_description="", research_goals="", preferences="",
             use_llm=True):
    """Generate experiment scaffold files. All user interaction (including
    the LLM interview) should happen before calling this."""
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

    # train.py — base template, then optionally customize with LLM
    if backend == "pt":
        train_code = TRAIN_PT.format(
            depth=depth, batch_size_pt=batch_size_pt,
        )
    else:
        train_code = TRAIN_MLX.format(
            depth=depth, batch_size_mlx=batch_size_mlx,
        )

    resolved_key = resolve_api_key(model_name, api_key) if use_llm else None

    if use_llm and resolved_key:
        print(f"\n  Customizing train.py with {model_name}...")
        try:
            train_code = customize_with_llm(
                train_code, "train.py", model_name, resolved_key,
                project_context, data_description, research_goals,
                preferences, backend, time_budget, depth,
            )
        except Exception as e:
            print(f"  ⚠ LLM customization failed for train.py: {e}")
            print("  → Using base template instead.")
    elif use_llm and not resolved_key:
        print("\n  ⚠ No API key found — skipping LLM customization, using base template.")

    with open(os.path.join(output_dir, "train.py"), "w") as f:
        f.write(train_code)

    # program.md — base template, then optionally customize with LLM
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

    if use_llm and resolved_key:
        print(f"  Customizing program.md with {model_name}...")
        try:
            program_md = customize_with_llm(
                program_md, "program.md", model_name, resolved_key,
                project_context, data_description, research_goals,
                preferences, backend, time_budget, depth,
            )
        except Exception as e:
            print(f"  ⚠ LLM customization failed for program.md: {e}")
            print("  → Using base template instead.")

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
            provider, env_var = _detect_provider(model_name)
            f.write(f"{env_var}={api_key}\n")

    # Initialize standalone git repo so experiments don't pollute the parent repo
    subprocess.run(["git", "init"], cwd=output_dir, capture_output=True)
    subprocess.run(["git", "add", "."], cwd=output_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"Scaffold: {tag}"],
        cwd=output_dir, capture_output=True,
    )

    return {
        "output_dir": output_dir,
        "prepare": os.path.join(output_dir, "prepare.py"),
        "train": os.path.join(output_dir, "train.py"),
        "program": os.path.join(output_dir, "program.md"),
    }


# ---------------------------------------------------------------------------
# Baseline validation — run prepare + train, parse results
# ---------------------------------------------------------------------------

def _detect_infra():
    """Detect hardware/OS info for the summary."""
    import subprocess
    info = {"os": platform.system(), "arch": platform.machine()}
    if platform.system() == "Darwin":
        try:
            chip = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            info["chip"] = chip
        except Exception:
            info["chip"] = platform.processor() or "Apple Silicon"
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True
            ).strip())
            info["memory_gb"] = round(mem_bytes / (1024**3))
        except Exception:
            info["memory_gb"] = "?"
    else:
        info["chip"] = platform.processor() or "unknown"
        info["memory_gb"] = "?"
    info["python"] = platform.python_version()
    return info


def _parse_train_output(output):
    """Parse the --- block from train.py output into a dict."""
    results = {}
    in_results = False
    for line in output.splitlines():
        line = line.strip()
        if line == "---":
            in_results = True
            continue
        if in_results and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            try:
                results[key] = float(val)
            except ValueError:
                results[key] = val
    return results


def run_baseline(output_dir, backend, time_budget, depth):
    """Run prepare.py + train.py in the scaffold dir, return parsed results or raise.

    The subprocess timeout is derived from the time_budget with 20% headroom
    on top of expected overhead (compilation, data loading, eval).
    """
    import subprocess

    print()
    print("=" * 60)
    print("  Running baseline validation...")
    print("=" * 60)

    # --- prepare ---
    print("\n  [1/2] prepare.py — downloading data & training tokenizer...")
    t0 = time.time()
    prep_result = subprocess.run(
        ["uv", "run", "prepare.py", "--num-shards", "10"],
        cwd=output_dir, capture_output=True, text=True, timeout=600,
    )
    prep_wall = time.time() - t0
    if prep_result.returncode != 0:
        print(f"\n  FAIL: prepare.py exited with code {prep_result.returncode}")
        print("  stderr:")
        for line in prep_result.stderr.strip().splitlines()[-20:]:
            print(f"    {line}")
        raise RuntimeError("prepare.py failed — scaffold is broken")
    for line in prep_result.stdout.splitlines():
        if any(kw in line for kw in ["Data:", "Tokenizer:", "Done"]):
            print(f"    {line.strip()}")
    print(f"    (took {prep_wall:.0f}s)")

    # --- train ---
    # Timeout = (time_budget + generous overhead for compile/eval/startup) * 1.2
    overhead = max(prep_wall, 60)  # at least 60s for compilation + eval
    train_timeout = int((time_budget + overhead) * 1.2)
    print(f"\n  [2/2] train.py — running baseline ({time_budget}s budget, timeout {train_timeout}s)...")
    t0 = time.time()
    train_result = subprocess.run(
        ["uv", "run", "train.py"],
        cwd=output_dir, capture_output=True, text=True, timeout=train_timeout,
    )
    train_wall = time.time() - t0
    if train_result.returncode != 0:
        print(f"\n  FAIL: train.py exited with code {train_result.returncode}")
        error_lines = train_result.stderr.strip().splitlines()[-30:]
        stdout_lines = train_result.stdout.strip().splitlines()[-30:]
        print("  stderr (last 30 lines):")
        for line in error_lines:
            print(f"    {line}")
        print("  stdout (last 30 lines):")
        for line in stdout_lines:
            print(f"    {line}")
        error_text = "\n".join(error_lines + stdout_lines)
        raise RuntimeError(error_text)

    results = _parse_train_output(train_result.stdout)
    if "val_bpb" not in results:
        print("\n  FAIL: train.py ran but produced no val_bpb")
        stdout_lines = train_result.stdout.strip().splitlines()[-20:]
        print("  stdout (last 20 lines):")
        for line in stdout_lines:
            print(f"    {line}")
        raise RuntimeError("\n".join(stdout_lines))

    results["_prep_wall"] = prep_wall
    results["_train_wall"] = train_wall
    print(f"    val_bpb: {results['val_bpb']:.6f} (took {train_wall:.0f}s wall)")
    return results


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
    "6": ("litellm-proxy", "LiteLLM", "LITELLM_API_KEY"),
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
    default_model = os.environ.get("DEFAULT_MODEL", "")
    print()
    print("  Which LLM should run the experiments?")
    print("    1) claude-sonnet-4-20250514  (Anthropic)")
    print("    2) claude-opus-4-20250514    (Anthropic)")
    print("    3) gpt-4o                    (OpenAI)")
    print("    4) o3                        (OpenAI)")
    print("    5) deepseek-r1               (DeepSeek)")
    print("    6) LiteLLM proxy             (any model via LiteLLM)")
    print("    7) custom")
    if default_model:
        print(f"\n  DEFAULT_MODEL={default_model}")
    print()
    default_choice = "1"
    # If DEFAULT_MODEL is set, find matching preset or use it directly
    if default_model:
        for k, (m, _, _) in LLM_PRESETS.items():
            if m == default_model:
                default_choice = k
                break
        else:
            default_choice = default_model  # will fall through to custom
    model_choice = ask("Pick a number or enter model ID", default=default_choice)

    if model_choice in LLM_PRESETS:
        model_name, provider, env_var = LLM_PRESETS[model_choice]
        if model_choice == "6":
            # LiteLLM — ask for actual model ID to route through proxy
            model_name = ask("Model ID to send to LiteLLM (e.g. gpt-4o, claude-sonnet-4-20250514, ollama/llama3)")
    elif model_choice == "7":
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
        "use_llm": True,
    }


def print_summary(output_dir, backend, model_name, time_budget, depth,
                   baseline_results=None, infra=None):
    backend_label = "PyTorch (CUDA)" if backend == "pt" else "MLX (Apple Silicon)"
    print()
    print("=" * 60)
    print(f"  Scaffold ready: {output_dir}/")
    print("=" * 60)

    # --- Infra ---
    if infra:
        print()
        print("  Hardware")
        print(f"    Chip:      {infra.get('chip', '?')}")
        print(f"    Memory:    {infra.get('memory_gb', '?')} GB")
        print(f"    OS:        {infra.get('os', '?')} ({infra.get('arch', '?')})")
        print(f"    Python:    {infra.get('python', '?')}")

    # --- Config ---
    print()
    print("  Config")
    print(f"    Backend:     {backend_label}")
    print(f"    Agent LLM:   {model_name}")
    print(f"    Time budget: {time_budget}s ({time_budget // 60} min)")
    print(f"    Depth:       {depth} layers")

    # --- Baseline ---
    if baseline_results:
        print()
        print("  Baseline (verified)")
        print(f"    val_bpb:          {baseline_results.get('val_bpb', '?')}")
        if "peak_vram_mb" in baseline_results:
            peak_gb = baseline_results["peak_vram_mb"] / 1024
            print(f"    peak_memory:      {peak_gb:.1f} GB")
        if "num_params_M" in baseline_results:
            print(f"    params:           {baseline_results['num_params_M']:.1f}M")
        if "num_steps" in baseline_results:
            print(f"    steps:            {int(baseline_results['num_steps'])}")
        if "training_seconds" in baseline_results:
            print(f"    training_time:    {baseline_results['training_seconds']:.0f}s")
        if "total_tokens_M" in baseline_results:
            print(f"    tokens:           {baseline_results['total_tokens_M']:.1f}M")

    # --- Files ---
    print()
    print("  Files")
    print(f"    prepare.py   ← data + tokenizer + eval (DO NOT MODIFY)")
    print(f"    train.py     ← model + training loop (agent modifies)")
    print(f"    program.md   ← experiment protocol (context + rules + loop)")

    # --- Next steps ---
    print()
    print(f"  Ready to go:")
    print(f"    cd {output_dir}")
    print(f"    claude --dangerously-skip-permissions")
    print()
    print(f"  Then paste:")
    print(f'    "Read program.md and start the experiment loop. Never stop."')
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
                        help="LLM for the agent (default: $DEFAULT_MODEL or claude-sonnet-4-20250514)")
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
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM customization, use base templates only")
    parser.add_argument("--no-interview", action="store_true",
                        help="Skip follow-up questions, go straight to scaffolding")

    args = parser.parse_args()

    # If --output-dir not given, go fully interactive
    if args.output_dir is None:
        config = interactive_setup()
    else:
        # One-shot mode — ask for missing context interactively
        print()
        print("=" * 60)
        print("  autoresearch-gen — scaffold a new experiment")
        print("=" * 60)

        project_context = args.context
        if project_context is None:
            project_context = ask_long(
                "Tell me about your project.",
                "What are you building or researching? Any relevant background the agent should know.",
            )

        data_description = args.data
        if data_description is None:
            data_description = ask_long(
                "What data are you working with?",
                "Where does it come from? How big is it? What format?",
            )

        research_goals = args.goals
        if research_goals is None:
            research_goals = ask_long(
                "What should the agent try to optimize or explore?",
                "e.g. 'minimize val_bpb', 'try different attention patterns'",
            )

        preferences = args.prefs
        if preferences is None:
            preferences = ask_long(
                "Any preferences for how the scaffold should start?",
                "e.g. 'start small and simple', 'use a 4-layer model first'",
            )

        config = {
            "output_dir": args.output_dir,
            "backend": args.backend or detect_backend(),
            "tag": args.tag or os.path.basename(os.path.normpath(args.output_dir)),
            "model_name": args.model or os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-20250514"),
            "api_key": args.api_key,
            "time_budget": args.time_budget or 300,
            "depth": args.depth or 8,
            "batch_size_pt": args.batch_size_pt,
            "batch_size_mlx": args.batch_size_mlx,
            "project_context": project_context or "",
            "data_description": data_description or "",
            "research_goals": research_goals or "",
            "preferences": preferences or "",
            "use_llm": not args.no_llm,
        }

    # --- LLM interview: enrich context before scaffolding ---
    if config["use_llm"] and not args.no_interview:
        resolved_key = resolve_api_key(config["model_name"], config.get("api_key"))
        if resolved_key:
            enriched = gather_deep_context(
                config["model_name"], resolved_key,
                config["project_context"], config["data_description"],
                config["research_goals"], config["preferences"],
                config["backend"],
            )
            config["project_context"] = enriched["project_context"]
            config["data_description"] = enriched["data_description"]
            config["research_goals"] = enriched["research_goals"]
            config["preferences"] = enriched["preferences"]

    files = generate(**config)

    # --- Always run baseline to guarantee a working scaffold ---
    infra = _detect_infra()
    used_llm = config["use_llm"]
    max_fix_attempts = 3
    baseline_results = None

    for attempt in range(max_fix_attempts + 1):
        try:
            baseline_results = run_baseline(
                config["output_dir"], config["backend"],
                config["time_budget"], config["depth"],
            )
            break
        except RuntimeError as e:
            error_output = str(e)
            if not used_llm or attempt == max_fix_attempts:
                raise
            resolved_key = resolve_api_key(config["model_name"], config.get("api_key"), interactive=False)
            if not resolved_key:
                raise
            print(f"\n  Attempt {attempt + 1}/{max_fix_attempts} failed. Sending error to LLM for fix...")
            train_path = os.path.join(config["output_dir"], "train.py")
            with open(train_path) as f:
                broken_code = f.read()
            try:
                fixed_code = fix_with_llm(
                    broken_code, error_output,
                    config["model_name"], resolved_key, config["backend"],
                )
                with open(train_path, "w") as f:
                    f.write(fixed_code)
                # Commit the fix
                subprocess.run(
                    ["git", "add", "train.py"], cwd=config["output_dir"], capture_output=True,
                )
                subprocess.run(
                    ["git", "commit", "-m", f"Auto-fix attempt {attempt + 1}"],
                    cwd=config["output_dir"], capture_output=True,
                )
                print(f"  LLM produced a fix. Retrying baseline...")
            except Exception as fix_err:
                print(f"  LLM fix failed: {fix_err}")
                raise RuntimeError(error_output) from fix_err

    print_summary(
        config["output_dir"], config["backend"], config["model_name"],
        config["time_budget"], config["depth"],
        baseline_results=baseline_results, infra=infra,
    )


if __name__ == "__main__":
    main()
