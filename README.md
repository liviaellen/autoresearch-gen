# autoresearch-gen

Scaffold generator for [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Tell it what you're working on, pick your LLM and backend, and it generates a ready-to-run experiment directory with `prepare.py`, `train.py`, and `program.md`.

Supports **PyTorch (CUDA)** and **MLX (Apple Silicon)**. Switch the agent LLM by passing a model name and key. Comes with a **Streamlit dashboard** to visualize any experiment's results.

## What it generates

```
experiments/my-run/
  prepare.py      ← data download, tokenizer, dataloader, eval (DO NOT MODIFY)
  train.py        ← model + training loop (agent modifies this)
  program.md      ← your context + experiment protocol + autonomous loop rules
  pyproject.toml  ← dependencies
  .gitignore
  .env            ← API key (if provided)
```

The `program.md` captures everything you told the generator — project context, data, goals, preferences — so the agent knows *why* it's running experiments, not just how.

## Prerequisites

Just [uv](https://docs.astral.sh/uv/). It handles Python, virtual envs, and all dependencies per experiment — nothing else to install.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **Mac (Apple Silicon):** auto-detects `--backend mlx`
- **GPU (CUDA):** pass `--backend pt`
- **No flag?** picks `mlx` on Apple Silicon, `pt` everywhere else

## Quick start

```bash
# Interactive — walks you through it
python gen.py

# Or one-shot with flags
python gen.py --output-dir experiments/mar11 --backend mlx
```

The interactive mode asks:

1. **Tell me about your project** — free text background
2. **What data are you working with?** — source, format, size
3. **What should the agent optimize?** — research goals
4. **Preferences for the starter code?** — how the scaffold should look
5. Tag, output dir, backend, LLM, API key, time budget, depth

Then generates everything and tells you what to do next.

## Dashboard

Streamlit dashboard to visualize experiment results. Auto-detects any metric from the TSV columns, figures out direction (lower/higher is better), and shows status.

```bash
# Launch — auto-discovers all experiments in experiments/
uv run --with streamlit --with pandas --with plotly streamlit run dashboard.py

# Or point at a specific experiment
uv run --with streamlit --with pandas --with plotly streamlit run dashboard.py -- --exp experiments/scaling-depth
```

What you get:
- **KPI cards** — best metric, baseline, improvement %, keep rate
- **Metric progression** — interactive chart with keep/revert markers
- **Best-so-far line** — running best across experiments
- **Status distribution** — pie chart of keep/revert/baseline
- **Overfitting analysis** — train vs eval gap (when available)
- **Auto-generated insights** — biggest win, worst attempt, diminishing returns
- **Full experiment log** — filterable table with highlighted best values

Works with any metric — val_bpb, cv_rmse, f1, accuracy, whatever your experiment tracks. The agent writes two TSV files: `experiments.tsv` (every iteration including reverts/crashes) and `results.tsv` (only baseline + kept experiments). The dashboard reads both.

## Examples

Four ready-to-generate examples. The first two show how little you need to know — just describe your goal and data. The last two show what's possible when you know your hardware and have a specific hypothesis.

### "I don't know much, just make it good"

You have data, you have a goal, you don't need to be an expert.

**Example 1: Just make it better**

```bash
python gen.py \
  --output-dir experiments/just-make-it-better \
  --backend mlx \
  --model claude-sonnet-4-20250514 \
  --context "I'm new to LLM pretraining. Just want to see how good a small model can get." \
  --data "climbmix-400b-shuffle from HuggingFace" \
  --goals "Get the lowest val_bpb you can. Try whatever you think will help."
```

No `--prefs`, no `--depth` override. The agent decides what to try.

**Example 2: I know my data, figure out the rest**

```bash
python gen.py \
  --output-dir experiments/find-what-works \
  --backend mlx \
  --model claude-sonnet-4-20250514 \
  --context "I have a dataset I want to pretrain on. Not sure what architecture or hyperparams to use." \
  --data "climbmix-400b-shuffle, 10 shards, ~60M tokens per shard" \
  --goals "Minimize val_bpb. Figure out what model size and settings work best for this data."
```

You described the data in detail but left the research strategy to the agent.

### "I know what I'm doing, here's the plan"

You know your infra, you have a hypothesis, you want the agent to execute a specific research plan.

**Example 3: Scale model depth on Apple Silicon**

```bash
python gen.py \
  --output-dir experiments/scaling-depth \
  --backend mlx \
  --model claude-sonnet-4-20250514 \
  --context "How deep can we scale a GPT on Apple Silicon with 128GB unified memory? \
Start small, push model size aggressively." \
  --data "climbmix-400b-shuffle from HuggingFace, 10 shards for quick iteration" \
  --goals "Minimize val_bpb. Scale depth and width to find the largest model \
that trains in 5 minutes." \
  --prefs "Start with 4 layers. Increase aggressively — try depths 4, 8, 12, 16, 20. \
Also try wider models (larger aspect ratio)." \
  --depth 4
```

**Example 4: LR schedule ablation**

```bash
python gen.py \
  --output-dir experiments/lr-schedule-ablation \
  --backend mlx \
  --model claude-sonnet-4-20250514 \
  --context "Systematic ablation of learning rate schedules. Fixed 8-layer GPT, \
only changing LR code." \
  --data "climbmix-400b-shuffle, 10 shards" \
  --goals "Find the LR schedule that minimizes val_bpb. Try cosine, linear, constant, \
cyclic, 1cycle. Keep architecture frozen." \
  --prefs "Do NOT change model architecture. Only modify LR code. Try at least 10 variants." \
  --depth 8
```

### Running an experiment

```bash
# 1. Generate the scaffold
python gen.py --output-dir experiments/my-run --backend mlx ...

# 2. Download data + train tokenizer (one-time, ~2 min)
cd experiments/my-run
uv run prepare.py

# 3. Run baseline to verify everything works
uv run train.py

# 4. Point your agent at it
# In Claude Code:
#   "Read program.md and let's kick off a new experiment. Do the setup first."

# 5. Watch results in another terminal
streamlit run ../../dashboard.py
```

## Use cases

### Overnight research on your Mac

```bash
python gen.py \
  --output-dir experiments/overnight \
  --backend mlx \
  --model claude-sonnet-4-20250514 \
  --context "Small GPT pretraining on Apple Silicon" \
  --data "climbmix-400b-shuffle, 10 shards" \
  --goals "Minimize val_bpb. Try different model sizes, LR schedules, and activations"
```

Wake up to ~100 experiments completed. The agent logs every iteration to `experiments.tsv` and curated keeps to `results.tsv`. View them with `streamlit run dashboard.py`.

### CUDA box with OpenAI agent

```bash
python gen.py \
  --output-dir experiments/h100-run \
  --backend pt \
  --model gpt-4o \
  --api-key sk-... \
  --context "8-layer GPT on H100 80GB" \
  --data "climbmix-400b, full dataset" \
  --goals "Lowest val_bpb possible"
```

### Compare LLM agents

```bash
# Same experiment, different researchers
python gen.py --output-dir experiments/claude-run --model claude-sonnet-4-20250514 \
  --context "Free exploration" --goals "Lowest val_bpb possible"

python gen.py --output-dir experiments/gpt4o-run --model gpt-4o --api-key sk-... \
  --context "Free exploration" --goals "Lowest val_bpb possible"
```

Compare in the dashboard — it auto-discovers all experiments.

### Interactive — just exploring

```bash
python gen.py
```

Walks you through everything step by step.

## Switching LLMs

The `--model` flag accepts any model ID. The `--api-key` is auto-routed:

| Model contains | Env var written |
|---|---|
| `claude` or `anthropic` | `ANTHROPIC_API_KEY` |
| `gpt`, `o1`, `o3` | `OPENAI_API_KEY` |
| anything else | `API_KEY` |

```bash
# Anthropic
python gen.py --output-dir exp/run1 --model claude-sonnet-4-20250514 --api-key sk-ant-...

# OpenAI
python gen.py --output-dir exp/run1 --model gpt-4o --api-key sk-...

# DeepSeek
python gen.py --output-dir exp/run1 --model deepseek-r1 --api-key sk-...

# Or set env var and skip --api-key
export ANTHROPIC_API_KEY=sk-ant-...
python gen.py --output-dir exp/run1 --model claude-sonnet-4-20250514
```

## All flags

```
python gen.py [OPTIONS]

Options:
  --output-dir DIR      Output directory (if omitted, goes interactive)
  --backend {pt,mlx}    Training backend (default: auto-detect)
  --tag TAG             Experiment tag (default: dir name)
  --model MODEL         LLM model ID (default: claude-sonnet-4-20250514)
  --api-key KEY         API key for the LLM
  --time-budget SECS    Training time budget (default: 300)
  --depth N             Transformer layers (default: 8)
  --batch-size-pt N     Batch size for PyTorch (default: 64)
  --batch-size-mlx N    Batch size for MLX (default: 16)
  --context TEXT        Project context — what you're building
  --data TEXT           Data description — source, format, size
  --goals TEXT          Research goals — what to optimize
  --prefs TEXT          Scaffold preferences — starter code style
```

## Project structure

```
gen.py              ← scaffold generator (single file)
dashboard.py        ← experiment tracking dashboard (Streamlit)
pyproject.toml      ← project deps
experiments/        ← generated experiment directories
```

## What we add on top of Karpathy's autoresearch

[Karpathy's autoresearch](https://github.com/karpathy/autoresearch) is a single repo with one `prepare.py`, one `train.py`, and one `program.md`. You clone it and point your agent at it.

**autoresearch-gen** adds:

| | autoresearch | autoresearch-gen |
|---|---|---|
| Setup | Clone repo, manually edit `program.md` | Run `gen.py`, describe your project, done |
| Backend | PyTorch only | **PyTorch + MLX** (Apple Silicon) |
| Context | You write `program.md` by hand | Generator bakes your context, data, goals, preferences into `program.md` |
| Agent LLM | Hardcoded | **Switch with `--model`** — Claude, GPT-4o, DeepSeek, anything |
| Logging | Single TSV | **Two TSVs** — `experiments.tsv` (all iterations) + `results.tsv` (keeps only) |
| Visualization | None | **Streamlit dashboard** — auto-detects metrics, shows progression, insights |
| Experiments | One at a time | **Experiment factory** — generate as many as you want, dashboard discovers all |

Think of it as:
- **autoresearch** = the experiment
- **autoresearch-gen** = the experiment factory + dashboard

## Attribution

Built on top of:

- [autoresearch](https://github.com/karpathy/autoresearch) by [@karpathy](https://github.com/karpathy) — the original autonomous pretraining research framework
- [autoresearch-mlx](https://github.com/karpathy/autoresearch/tree/master) — MLX backend for Apple Silicon, community-contributed

## Maintainer

[@liviaellen](https://github.com/liviaellen) at [Mem0](https://github.com/mem0ai) · [@ellen_in_sf](https://x.com/ellen_in_sf)

## License

MIT
