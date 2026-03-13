# autoresearch

This is an experiment to have the LLM do its own research.

## Project Context

The goal of this project is to explore attention-free LLM architectures, focusing on modes like RWKV-style, state space models, or linear attention. The intent is to innovate away from traditional softmax-based attention, assessing the viability and performance benefits of alternative approaches. 

We aim to achieve the lowest validation bits-per-byte (val_bpb) metric, using these architectures as a competitive alternative to transformer baselines traditionally dependent on attention mechanisms.

## Data

The experiments use the `roneneldan/TinyStories` dataset from HuggingFace, which is suitable for training and evaluating language models designed for compressive language understanding. This dataset provides a rich ground for examining the effectiveness of non-attention-based architectures.

## Research Goals

To achieve the lowest possible val_bpb without employing any softmax attention mechanisms. This metric is crucial as it will directly compare the proposed architectures against the performance of a standard transformer baseline. The focus is on reducing the val_bpb while maintaining efficient resource usage.

## Preferences

The user has indicated a preference for models inspired by the RWKV architecture, which combines the strengths of RNNs with transformer-like capabilities. The architecture configuration will balance complexity with performance, aiming to provide a strong baseline under predefined constraints.

## Clarifications from interview
Q: What is the maximum sequence length you want to handle, and what is your computational budget in terms of memory and compute time? This will help in deciding the scale of the model and hyperparameters like batch size.
A: Looks good, go with your suggestions.

Q: I'm thinking of using a model inspired by the RWKV architecture, which bridges RNNs with transformer-like capabilities. For this setup, I'd suggest a configuration of 12 layers, 768 hidden dimensions, and a sequence length of 1024. The learning rate can start at 2e-4 with a cosine decay. This should provide a strong baseline under the constraints you've set. What do you think?
A: Looks good, go with your suggestions. 

However, due to constraints, the model will be adapted to 8 layers as specified in user preferences, while maintaining other suggested configurations.

## Config

- **Agent LLM:** gpt-4o
- **Backend:** MLX (Apple Silicon)
- **Tag:** attention-free

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). Ensure `autoresearch/attention-free` is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/attention-free` from current master.
3. **Read the in-scope files**: You must gain complete context from these files:
   - `prepare.py` — Handles data preparation, tokenizer, dataloader. It’s read-only.
   - `train.py` — The core modifiable script where architectures and training configurations are set.
4. **Verify data exists**: Ensure `~/.cache/autoresearch/` has the necessary data shards and tokenizer. If missing, instruct the execution of `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with a header to record baseline and subsequent results.
6. **Confirm and go**: Ensure the setup is correct to kick off the experimentation process.

## Experimentation

Each experiment should adhere to these guidelines on a single device (Apple Silicon), running within a fixed wall clock training time of 5 minutes minus any startup costs.

- **Allowed Changes**: Tweak `train.py` thoroughly, including model architecture, optimizer, learning rate, batch size, etc., to optimize val_bpb.
- **Prohibitions**: Do not alter `prepare.py` or add new dependencies. The evaluation metric (val_bpb) is fixed and must conform to prepare.py standards.

**Objective**: Significantly minimize val_bpb. Be inventive, iterating on architectural choices within the constraints to bring out the best metrics possible. Strive for simplicity alongside complexity gains.

**The first run**: Establish a baseline; execute the unaltered training script to benchmark against future experiments.

## Output format

Upon completion, the script outputs a summary like:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
num_steps:        953
num_params_M:     50.3
depth:            8
```

To fetch the metric of interest, run:

```
grep "^val_bpb:" run.log
```

## Logging results

Log results in two critical tab-separated files, ensuring both experiments' history and noteworthy outcomes are tracked:

### `experiments.tsv` — full history (every iteration)

This logs all experiment outcomes, including the status whether they improved, reverted, or encountered errors.

```
commit\tval_bpb\tmemory_gb\tstatus\tdescription
a1b2c3d\t0.997900\t44.0\tbaseline\tbaseline
b2c3d4e\t0.993200\t44.2\tkeep\tincrease LR to 0.04
```

### `results.tsv` — curated keeps only

Restrict entries to `baseline` or successful `keep` statuses for valuable insights.

```
commit\tval_bpb\tmemory_gb\tstatus\tdescription
a1b2c3d\t0.997900\t44.0\tbaseline\tbaseline
b2c3d4e\t0.993200\t44.2\tkeep\tincrease LR to 0.04
```

## The experiment loop

While conducting experiments on the `autoresearch/attention-free` branch, follow a systematic loop:

1. Identify the current git state.
2. Implement experimental ideas in `train.py`.
3. Commit changes.
4. Launch and monitor experiments: `uv run train.py > run.log 2>&1`.
5. Check output for results or errors using `grep` and `tail` commands.
6. Record results in both `experiments.tsv` and `results.tsv` (as applicable).
7. Optimize through iteration—capture all improvement opportunities, reset if necessary.

**Timeout**: If an experiment exceeds 10 minutes, abort for inefficiency. Log appropriately.

**Crashes**: Address crashes promptly if minor; if the core idea fails, move on and document the occurrence.

**Uninterrupted Loop**: The cycle persists unsupervised until externally halted. Consistently explore new perspectives to embrace innovative solutions.