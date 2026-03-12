.PHONY: help gen dashboard diagram baseline agent clean

# Default experiment dir (override with EXP=experiments/foo)
EXP ?= experiments/my-run

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Scaffold generation
# ---------------------------------------------------------------------------

gen: ## Generate scaffold — interactive, or pass EXP/CONTEXT/DATA/GOALS
ifdef CONTEXT
	python gen.py --output-dir $(EXP) --context "$(CONTEXT)" --data "$(DATA)" --goals "$(GOALS)"
else
	python gen.py
endif

# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

dashboard: ## Launch Streamlit dashboard (set EXP to scope to one experiment)
ifdef EXP_SCOPE
	uv run --with streamlit --with pandas --with plotly streamlit run dashboard.py -- --exp $(EXP_SCOPE)
else
	uv run --with streamlit --with pandas --with plotly streamlit run dashboard.py
endif

# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

diagram: ## Generate Excalidraw diagram for experiment (set EXP)
	python excalidraw_gen.py $(EXP) $(if $(OUT),-o $(OUT))

# ---------------------------------------------------------------------------
# Running experiments
# ---------------------------------------------------------------------------

baseline: ## Run baseline in experiment dir (set EXP)
	cd $(EXP) && uv run prepare.py --num-shards 10 && uv run train.py

agent: ## Launch Claude agent in experiment dir (set EXP)
	cd $(EXP) && claude --dangerously-skip-permissions

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

clean: ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
