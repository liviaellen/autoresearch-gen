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
commit\tval_bpb\tmemory_gb\tstatus\tdescription
a1b2c3d\t0.997900\t44.0\tbaseline\tbaseline
b2c3d4e\t0.993200\t44.2\tkeep\tincrease LR to 0.04
c3d4e5f\t1.005000\t44.0\trevert\tswitch to GeLU activation
d4e5f6g\t0.000000\t0.0\tcrash\tdouble model width (OOM)
```

### `results.tsv` — curated keeps only

Only `baseline` and `keep` rows go here. This is what you build on.

```
commit\tval_bpb\tmemory_gb\tstatus\tdescription
a1b2c3d\t0.997900\t44.0\tbaseline\tbaseline
b2c3d4e\t0.993200\t44.2\tkeep\tincrease LR to 0.04
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
