.PHONY: help gen dashboard diagram baseline agent test clean

# Print each command before executing and stop on errors;
# this ensures failing recipes show the command that ran.
.SHELLFLAGS := -exu -o pipefail


# Default experiment dir (override with EXP=experiments/foo)
EXP ?= experiments/my-run

help: ## Show available targets
	@echo "example: make help"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Scaffold generation
# ---------------------------------------------------------------------------

gen: ## Generate scaffold — interactive, or pass EXP/CONTEXT/DATA/GOALS
	@echo "example: make gen CONTEXT=\"some text\" EXP=experiments/foo"
	@echo "  EXP      — output directory          (e.g. experiments/mar10)"
	@echo "  CONTEXT  — project description        (free text string)"
	@echo "  DATA     — dataset description         (e.g. 'climbmix-400b, 6.5K parquet shards')"
	@echo "  GOALS    — research goals              (e.g. 'minimize val_bpb')"
	@echo "  BACKEND  — training backend            (pt | mlx, default: auto-detect)"
	@echo "  MODEL    — LLM model ID               (e.g. claude-sonnet-4-20250514, gpt-4o)"
	@echo "  PREFS    — scaffold preferences        (e.g. 'start small, 4 layers')"
	@echo "  DEPTH    — transformer layers          (integer, default: 8)"
ifdef CONTEXT
	python gen.py --output-dir $(EXP) \
		--context "$(CONTEXT)" \
		$(if $(DATA),--data "$(DATA)") \
		$(if $(GOALS),--goals "$(GOALS)") \
		$(if $(BACKEND),--backend $(BACKEND)) \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(PREFS),--prefs "$(PREFS)") \
		$(if $(DEPTH),--depth $(DEPTH))
else
	python gen.py
endif

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

dashboard: ## Launch Streamlit dashboard (set EXP_SCOPE to scope to one experiment)
	@echo "example: make dashboard EXP_SCOPE=experiments/foo"
	@echo "  EXP_SCOPE — experiment directory to scope to (optional, e.g. experiments/mar10)"
ifdef EXP_SCOPE
	uv run --with streamlit --with pandas --with plotly streamlit run dashboard.py -- --exp $(EXP_SCOPE)
else
	uv run --with streamlit --with pandas --with plotly streamlit run dashboard.py
endif

# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

diagram: ## Generate Excalidraw diagram + PNG for experiment (set EXP)
	@echo "example: make diagram EXP=experiments/foo"
	@echo "  EXP — experiment directory            (e.g. experiments/attention-free)"
	python excalidraw_gen.py $(EXP)

# ---------------------------------------------------------------------------
# Running experiments
# ---------------------------------------------------------------------------

baseline: ## Run baseline in experiment dir (set EXP)
	@echo "example: make baseline EXP=experiments/foo"
	@echo "  EXP — experiment directory            (e.g. experiments/attention-free)"
	cd $(EXP) && uv run prepare.py --num-shards 10 && uv run train.py

agent: ## Launch Claude agent in experiment dir (set EXP)
	@echo "example: make agent EXP=experiments/foo"
	@echo "  EXP — experiment directory            (e.g. experiments/attention-free)"
	cd $(EXP) && claude --dangerously-skip-permissions

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test: ## Run tests
	uv run --extra test pytest tests/ -v

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

clean: ## Remove __pycache__ and .pyc files
	@echo "example: make clean"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
