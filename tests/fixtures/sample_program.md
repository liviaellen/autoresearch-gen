# autoresearch

This is an experiment to have the LLM do its own research.

## Project Context

Small GPT pretraining on Apple Silicon, exploring attention-free architectures.

## Data

Using roneneldan/TinyStories from HuggingFace.

## Research Goals

To achieve the lowest possible val_bpb without any softmax attention. Minimize val_bpb.

## Preferences

Start with RWKV-style architecture.

## Clarifications from interview
Q: What model size?
A: Looks good, go with your suggestions.

## Config

- **Agent LLM:** claude-sonnet-4-20250514
- **Backend:** MLX (Apple Silicon)
- **Tag:** attention-free

## Setup

To set up a new experiment, work with the user to:
