"""
Experiment Tracking Dashboard — auto-adapts to any experiment's metrics and structure.

Works with both autoresearch-gen (LLM pretraining) and autoresearch-lab (tabular ML).
Reads results.tsv / experiments.tsv, auto-detects metrics, direction, and status.

Usage:
    streamlit run dashboard.py
    streamlit run dashboard.py -- --exp experiments/overnight-m5
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Config ──────────────────────────────────────────────────────────────────

EXPERIMENTS_DIR = Path("experiments")
# Metrics where higher = better
HIGHER_IS_BETTER = {"auc", "f1", "accuracy", "precision", "recall", "r2"}
# Metrics where lower = better
LOWER_IS_BETTER = {"rmse", "mse", "mae", "mape", "loss", "error", "logloss", "log_loss", "bpb", "perplexity"}


def metric_direction(col_name: str) -> str:
    """Infer whether a metric column is 'lower' or 'higher' is better."""
    low = col_name.lower()
    for kw in HIGHER_IS_BETTER:
        if kw in low:
            return "higher"
    for kw in LOWER_IS_BETTER:
        if kw in low:
            return "lower"
    return "lower"  # default assumption


def is_metric_col(col: str, df: pd.DataFrame) -> bool:
    """Heuristic: numeric column that isn't an index/commit/status."""
    skip = {"commit", "status", "description", "_idx"}
    if col.lower() in skip:
        return False
    return pd.api.types.is_numeric_dtype(df[col])


def parse_program_md(path: Path) -> dict:
    """Extract metadata from program.md — handles both lab and gen formats."""
    info: dict = {}
    if not path.exists():
        return info
    text = path.read_text()

    # --- autoresearch-lab format ---
    m = re.search(r"\*\*Target column:\*\*\s*`([^`]+)`", text)
    if m:
        info["target"] = m.group(1)
    m = re.search(r"\*\*Task:\*\*\s*(\w+)", text)
    if m:
        info["task"] = m.group(1)
    m = re.search(r"\*\*Metric:\*\*\s*(.+)", text)
    if m:
        info["metric_desc"] = m.group(1).strip()
    m = re.search(r"\*\*Goal:\*\*\s*(?:Minimize|Maximize)\s*`([^`]+)`", text)
    if m:
        info["primary_metric"] = m.group(1)

    # --- autoresearch-gen format ---
    m = re.search(r"\*\*Backend:\*\*\s*(.+)", text)
    if m:
        info["backend"] = m.group(1).strip()
    m = re.search(r"\*\*Agent LLM:\*\*\s*(.+)", text)
    if m:
        info["agent_llm"] = m.group(1).strip()
    m = re.search(r"\*\*Tag:\*\*\s*(.+)", text)
    if m:
        info["tag"] = m.group(1).strip()

    # Project context (gen format)
    m = re.search(r"## Project Context\s+(.+?)(?=\n## )", text, re.DOTALL)
    if m:
        info["context"] = m.group(1).strip()

    # Research goals (gen format)
    m = re.search(r"## Research Goals\s+(.+?)(?=\n## )", text, re.DOTALL)
    if m:
        info["goals"] = m.group(1).strip()

    # Detect val_bpb as primary metric for pretraining experiments
    if "val_bpb" in text and "primary_metric" not in info:
        info["primary_metric"] = "val_bpb"
        info["task"] = info.get("task", "pretraining")
        info["metric_desc"] = info.get("metric_desc", "Bits per byte (lower is better)")

    # Direction
    if "lowest val_bpb" in text.lower() or "minimize" in text.lower():
        info["direction"] = "lower"
    elif "maximize" in text.lower():
        info["direction"] = "higher"

    return info


def discover_experiments() -> list[Path]:
    """Find experiment directories that have at least a program.md."""
    if not EXPERIMENTS_DIR.exists():
        return []
    return sorted(
        [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir() and (d / "program.md").exists()],
        key=lambda p: p.name,
    )


def load_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    # Replace "-" with NaN
    df = df.replace("-", pd.NA)
    # Try to convert numeric columns
    for col in df.columns:
        if col not in ("commit", "status", "description"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Experiment Tracker", layout="wide", page_icon=":microscope:")

# ── Auto-refresh ───────────────────────────────────────────────────────────

REFRESH_INTERVALS = {"Off": 0, "5s": 5, "10s": 10, "30s": 30, "60s": 60}


def _file_mtime(path: Path) -> float:
    """Get file modification time, 0 if missing."""
    try:
        return path.stat().st_mtime
    except (FileNotFoundError, OSError):
        return 0.0


# ── Sidebar: links ─────────────────────────────────────────────────────────

st.sidebar.markdown(
    '<a href="https://github.com/liviaellen/autoresearch-gen" target="_blank">'
    '<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" width="20" style="vertical-align:middle"> '
    "liviaellen/autoresearch-gen</a>"
    " &nbsp; "
    '<a href="https://x.com/ellen_in_sf" target="_blank">'
    '<img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18" style="vertical-align:middle"> '
    "@ellen_in_sf</a>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ── Sidebar: experiment picker ──────────────────────────────────────────────

experiments = discover_experiments()

if not experiments:
    st.error("No experiments found in `experiments/` directory.")
    st.stop()

exp_names = [e.name for e in experiments]
selected_name = st.sidebar.selectbox("Experiment", exp_names, index=0)
exp_dir = EXPERIMENTS_DIR / selected_name

# Load metadata
meta = parse_program_md(exp_dir / "program.md")

# Load data
experiments_df = load_tsv(exp_dir / "experiments.tsv")
results_df = load_tsv(exp_dir / "results.tsv")

# Pick the best available dataframe
df = experiments_df if experiments_df is not None else results_df

# ── Auto-refresh controls ──────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Live refresh**")
refresh_label = st.sidebar.select_slider("Interval", options=list(REFRESH_INTERVALS.keys()), value="Off")
refresh_secs = REFRESH_INTERVALS[refresh_label]

if st.sidebar.button("Refresh now"):
    st.rerun()

# Track file modification times for smart refresh
_exp_mtime = _file_mtime(exp_dir / "experiments.tsv")
_res_mtime = _file_mtime(exp_dir / "results.tsv")

# Auto-rerun on interval
if refresh_secs > 0:
    # Store last known mtimes in session state
    mtime_key = f"_mtime_{selected_name}"
    last_mtimes = st.session_state.get(mtime_key, (0.0, 0.0))
    st.session_state[mtime_key] = (_exp_mtime, _res_mtime)

    # Show countdown
    placeholder = st.sidebar.empty()
    placeholder.caption(f"Next refresh in {refresh_secs}s")
    time.sleep(refresh_secs)
    st.rerun()

st.sidebar.markdown("---")
if meta.get("target"):
    st.sidebar.markdown(f"**Target:** `{meta['target']}`")
st.sidebar.markdown(f"**Task:** {meta.get('task', 'unknown')}")
st.sidebar.markdown(f"**Metric:** {meta.get('metric_desc', 'see program.md')}")
if meta.get("backend"):
    st.sidebar.markdown(f"**Backend:** {meta['backend']}")
if meta.get("agent_llm"):
    st.sidebar.markdown(f"**Agent LLM:** {meta['agent_llm']}")
if meta.get("context"):
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Context**")
    st.sidebar.markdown(
        f'<div style="max-height:240px;overflow-y:auto;padding:8px;'
        f'background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);'
        f'border-radius:6px;font-size:13px;line-height:1.5">'
        f'{meta["context"]}</div>',
        unsafe_allow_html=True,
    )

if df is None:
    st.title(f"{selected_name}")
    st.warning("No results.tsv or experiments.tsv found yet — run some experiments first!")
    st.markdown("#### Experiment config")
    st.json(meta)
    st.stop()

# ── Auto-detect metrics ─────────────────────────────────────────────────────

metric_cols = [c for c in df.columns if is_metric_col(c, df)]
primary = meta.get("primary_metric")
if primary and primary not in metric_cols:
    primary = None
if not primary and metric_cols:
    # Pick the first cv_ column, or just the first metric
    cv_cols = [c for c in metric_cols if c.startswith("cv_")]
    primary = cv_cols[0] if cv_cols else metric_cols[0]

direction = meta.get("direction") or (metric_direction(primary) if primary else "lower")
better_fn = max if direction == "higher" else min
worse_fn = min if direction == "higher" else max
arrow = "higher" if direction == "higher" else "lower"

has_status = "status" in df.columns
has_description = "description" in df.columns

# Add index column
df = df.reset_index(drop=True)
df.insert(0, "#", range(1, len(df) + 1))

# ── Header ──────────────────────────────────────────────────────────────────

st.title(f"{selected_name}")
subtitle_parts = []
if meta.get("target"):
    subtitle_parts.append(f"Target: **{meta['target']}**")
if meta.get("task"):
    subtitle_parts.append(f"Task: **{meta['task']}**")
if primary:
    subtitle_parts.append(f"Primary metric: **{primary}** ({arrow} is better)")
subtitle_parts.append(f"**{len(df)}** experiments")
st.markdown(" · ".join(subtitle_parts))

# ── KPI cards ───────────────────────────────────────────────────────────────

valid = df[primary].dropna()
if len(valid) > 0:
    best_val = better_fn(valid)
    best_row = df.loc[df[primary] == best_val].iloc[0]
    baseline_val = valid.iloc[0] if len(valid) > 0 else None

    cols = st.columns(4)

    with cols[0]:
        st.metric(f"Best {primary}", f"{best_val:,.4f}")
        if has_description:
            st.caption(best_row.get("description", ""))

    with cols[1]:
        if baseline_val is not None:
            st.metric("Baseline", f"{baseline_val:,.4f}")
            if has_description:
                st.caption(df.iloc[0].get("description", ""))

    with cols[2]:
        if baseline_val and baseline_val != 0:
            abs_diff = abs(best_val - baseline_val)
            if direction == "lower":
                pct = (baseline_val - best_val) / baseline_val * 100
            else:
                pct = (best_val - baseline_val) / baseline_val * 100
            st.metric(
                "Improvement",
                f"{pct:.1f}%",
                delta=f"{abs_diff:,.4f} {'lower' if direction == 'lower' else 'higher'}",
                delta_color="normal",
            )

    with cols[3]:
        if has_status:
            keeps = df[df["status"].isin(["keep", "baseline"])].shape[0]
            reverts = df[df["status"] == "revert"].shape[0]
            total = keeps + reverts
            pct_kept = keeps / max(total, 1) * 100
            st.metric("Keep Rate", f"{pct_kept:.0f}%")
            st.caption(f"{keeps} kept / {reverts} reverted")

# ── Autotune Progress (combined scatter + running best) ────────────────────

if primary and has_status:
    st.markdown("---")

    kept_mask = df["status"].isin(["keep", "baseline"])
    discarded_mask = df["status"].isin(["revert", "crash"])
    kept_df = df[kept_mask].dropna(subset=[primary])
    discarded_df = df[discarded_mask].dropna(subset=[primary])

    # Compute running best from kept experiments
    running_best_rows = []
    best_so_far_val = None
    for _, row in df.iterrows():
        val = row[primary]
        if pd.isna(val):
            continue
        if row.get("status") in ("keep", "baseline"):
            if best_so_far_val is None or (direction == "lower" and val < best_so_far_val) or (direction == "higher" and val > best_so_far_val):
                best_so_far_val = val
                running_best_rows.append(row)

    n_kept = len(kept_df)
    n_total = len(df.dropna(subset=[primary]))

    fig_progress = go.Figure()

    # Discarded experiments — small gray dots
    if len(discarded_df) > 0:
        fig_progress.add_trace(go.Scatter(
            x=discarded_df["#"],
            y=discarded_df[primary],
            mode="markers",
            marker=dict(size=5, color="rgba(180,180,180,0.35)", line=dict(width=0)),
            name="Discarded",
            hovertemplate="#%{x}<br>" + primary + ": %{y:.4f}<br>%{customdata[0]}<extra>Discarded</extra>",
            customdata=discarded_df[["description"]].values if has_description else None,
        ))

    # Kept experiments — larger green dots
    if len(kept_df) > 0:
        fig_progress.add_trace(go.Scatter(
            x=kept_df["#"],
            y=kept_df[primary],
            mode="markers+text",
            marker=dict(size=10, color="#3fb950", line=dict(width=1, color="rgba(255,255,255,0.5)")),
            name="Kept",
            text=kept_df["description"].apply(lambda d: str(d)[:30] if pd.notna(d) else "") if has_description else None,
            textposition="top right",
            textfont=dict(size=9, color="rgba(80,80,80,0.85)"),
            hovertemplate="#%{x}<br>" + primary + ": %{y:.4f}<br>%{customdata[0]}<extra>Kept</extra>",
            customdata=kept_df[["description"]].values if has_description else None,
        ))

    # Running best step-line
    if running_best_rows:
        rb_df = pd.DataFrame(running_best_rows)
        fig_progress.add_trace(go.Scatter(
            x=rb_df["#"],
            y=rb_df[primary],
            mode="lines+markers",
            line=dict(color="#3fb950", width=2, shape="hv"),
            marker=dict(size=9, color="#3fb950", line=dict(width=1.5, color="white")),
            name="Running best",
            hovertemplate="#%{x}<br>" + primary + ": %{y:.4f}<extra>Best</extra>",
        ))

    fig_progress.update_layout(
        title=f"Autotune Progress: {n_total} Experiments, {n_kept} Kept Improvements",
        xaxis_title="Experiment #",
        yaxis_title=f"{primary} ({'lower' if direction == 'lower' else 'higher'} is better)",
        template="plotly_dark",
        height=520,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=12),
        ),
        hovermode="closest",
        margin=dict(t=80, b=60),
    )
    st.plotly_chart(fig_progress, use_container_width=True)

# ── Primary metric over experiments ─────────────────────────────────────────

st.markdown("---")

# Let user pick which metrics to chart
selected_metrics = st.multiselect(
    "Metrics to plot",
    metric_cols,
    default=[primary] if primary else metric_cols[:2],
)

if selected_metrics:
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, col in enumerate(selected_metrics):
        color = colors[i % len(colors)]

        if has_status:
            # Color points by status
            for status, marker_sym, opacity in [("keep", "circle", 1.0), ("baseline", "diamond", 1.0), ("revert", "x", 0.4)]:
                mask = df["status"] == status
                if mask.any():
                    fig.add_trace(go.Scatter(
                        x=df.loc[mask, "#"], y=df.loc[mask, col],
                        mode="markers",
                        marker=dict(symbol=marker_sym, size=7 if status != "revert" else 5, opacity=opacity, color=color),
                        name=f"{col} ({status})",
                        legendgroup=col,
                        hovertemplate=(
                            "#%{x}<br>" + col + ": %{y:,.2f}<br>"
                            + ("Desc: %{customdata[0]}" if has_description else "")
                        ),
                        customdata=df.loc[mask, ["description"]].values if has_description else None,
                    ))
            # Trend line (all points)
            fig.add_trace(go.Scatter(
                x=df["#"], y=df[col],
                mode="lines", line=dict(color=color, width=1, dash="dot"),
                opacity=0.4, showlegend=False, legendgroup=col,
                hoverinfo="skip",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=df["#"], y=df[col], mode="lines+markers",
                name=col, line=dict(color=color),
            ))

    # Add baseline reference line
    if primary in selected_metrics and baseline_val is not None:
        fig.add_hline(y=baseline_val, line_dash="dash", line_color="rgba(248,81,73,0.4)",
                      annotation_text="baseline", annotation_position="top left")

    fig.update_layout(
        title="Metric Progression",
        xaxis_title="Experiment #", yaxis_title="Score",
        template="plotly_dark", height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(t=80, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Best-so-far progression ────────────────────────────────────────────────

if primary and has_status:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Best-so-far Progression")
        running_best = []
        best_so_far = None
        for _, row in df.iterrows():
            val = row[primary]
            if pd.isna(val):
                continue
            if row.get("status") in ("keep", "baseline") or not has_status:
                if best_so_far is None or (direction == "lower" and val < best_so_far) or (direction == "higher" and val > best_so_far):
                    best_so_far = val
                    running_best.append(row)

        if running_best:
            prog_df = pd.DataFrame(running_best)
            fig2 = px.line(
                prog_df, x="#", y=primary, markers=True,
                hover_data=["description"] if has_description else None,
                template="plotly_dark",
            )
            fig2.update_traces(line_color="#3fb950", marker=dict(size=10))
            fig2.update_layout(height=350, yaxis_title=primary)
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Status Distribution")
        if has_status:
            status_counts = df[df["status"] != "baseline"]["status"].value_counts().reset_index()
            status_counts.columns = ["status", "count"]
            color_map = {"keep": "#3fb950", "revert": "#f85149"}
            fig3 = px.pie(
                status_counts, names="status", values="count",
                color="status", color_discrete_map=color_map,
                template="plotly_dark",
            )
            fig3.update_layout(height=350)
            st.plotly_chart(fig3, use_container_width=True)

# ── Overfitting analysis (if train metric exists) ──────────────────────────

train_cols = [c for c in metric_cols if "train" in c.lower()]
cv_cols = [c for c in metric_cols if "cv" in c.lower()]
val_cols = [c for c in metric_cols if "val" in c.lower() and "train" not in c.lower()]

if train_cols and (cv_cols or val_cols):
    st.markdown("---")
    st.subheader("Overfitting Analysis")

    train_col = train_cols[0]
    eval_col = cv_cols[0] if cv_cols else val_cols[0]
    gap_df = df[["#", train_col, eval_col]].dropna(subset=[train_col, eval_col]).copy()
    gap_df["gap"] = gap_df[eval_col] - gap_df[train_col]

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=gap_df["#"], y=gap_df[train_col], name=train_col, marker_color="rgba(63,185,80,0.5)"))
    fig4.add_trace(go.Bar(x=gap_df["#"], y=gap_df[eval_col], name=eval_col, marker_color="rgba(88,166,255,0.5)"))
    fig4.update_layout(
        barmode="group", template="plotly_dark", height=400,
        xaxis_title="Experiment #", yaxis_title="Score",
        title=f"Train vs Eval Gap (avg gap: {gap_df['gap'].mean():,.0f})",
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Gap by status
    if has_status:
        gap_full = df[["#", train_col, eval_col, "status"]].dropna(subset=[train_col, eval_col]).copy()
        gap_full["gap"] = gap_full[eval_col] - gap_full[train_col]
        gap_by_status = gap_full.groupby("status")["gap"].mean().reset_index()
        gap_by_status.columns = ["status", "avg_gap"]
        st.dataframe(gap_by_status.style.format({"avg_gap": "{:,.0f}"}), use_container_width=True)

# ── Insights (auto-generated) ──────────────────────────────────────────────

st.markdown("---")
st.subheader("Auto-generated Insights")

insights = []
if primary and len(valid) > 1:
    best_idx = df.loc[df[primary] == best_val, "#"].iloc[0]
    if has_description:
        best_desc = best_row.get("description", "")
        insights.append(f"**Best model** (#{best_idx}): {best_desc} — {primary} = **{best_val:,.2f}**")

    # Biggest single-step improvement among kept models
    if has_status:
        kept = df[df["status"].isin(["keep", "baseline"])].copy()
        if len(kept) > 1:
            kept = kept.sort_values("#")
            kept["delta"] = kept[primary].diff()
            if direction == "lower":
                best_step = kept.loc[kept["delta"].idxmin()]
                if pd.notna(best_step["delta"]):
                    insights.append(f"**Biggest single-step win**: #{int(best_step['#'])} — {primary} dropped by **{abs(best_step['delta']):,.0f}** ({best_step.get('description', '')})")
            else:
                best_step = kept.loc[kept["delta"].idxmax()]
                if pd.notna(best_step["delta"]):
                    insights.append(f"**Biggest single-step win**: #{int(best_step['#'])} — {primary} jumped by **{best_step['delta']:,.2f}** ({best_step.get('description', '')})")

        # Worst revert
        reverts = df[df["status"] == "revert"].dropna(subset=[primary])
        if len(reverts) > 0:
            if direction == "lower":
                wr = reverts.loc[reverts[primary].idxmax()]
            else:
                wr = reverts.loc[reverts[primary].idxmin()]
            insights.append(f"**Worst attempt**: #{int(wr['#'])} — {primary} = {wr[primary]:,.2f} ({wr.get('description', '')})")

        # Diminishing returns
        last_n = min(10, len(kept))
        if last_n > 2:
            recent = kept.tail(last_n)
            recent_range = abs(recent[primary].max() - recent[primary].min())
            total_range = abs(valid.max() - valid.min())
            if total_range > 0 and recent_range / total_range < 0.1:
                insights.append(f"**Diminishing returns**: Last {last_n} kept experiments span only **{recent_range:,.0f}** ({recent_range / total_range * 100:.1f}% of total range) — hitting a plateau.")

    # Overfit insight
    if train_cols:
        tc = train_cols[0]
        valid_train = df[[primary, tc]].dropna()
        if len(valid_train) > 0:
            avg_gap = (valid_train[primary] - valid_train[tc]).mean()
            insights.append(f"**Average overfit gap** ({primary} − {tc}): **{avg_gap:,.0f}**")

if not insights:
    insights.append("Run more experiments to generate insights.")

for ins in insights:
    st.markdown(f"- {ins}")

# ── Architecture Diagram ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("Architecture Diagram")

# Find existing diagram PNG
exp_name_slug = exp_dir.name
diagram_png = exp_dir / f"{exp_name_slug}.png"
diagram_excalidraw = exp_dir / f"{exp_name_slug}.excalidraw"
# Also check architecture.png as fallback
if not diagram_png.exists() and (exp_dir / "architecture.png").exists():
    diagram_png = exp_dir / "architecture.png"
    diagram_excalidraw = exp_dir / "architecture.excalidraw"

diagram_col1, diagram_col2 = st.columns([3, 1])

with diagram_col2:
    n_experiments = len(df)
    st.metric("Experiments", n_experiments)
    if diagram_png.exists():
        mtime = datetime.datetime.fromtimestamp(diagram_png.stat().st_mtime)
        st.caption(f"Last generated: {mtime:%Y-%m-%d %H:%M}")

    if st.button("Regenerate Diagram"):
        with st.spinner("Generating diagram..."):
            try:
                from excalidraw_gen import generate_diagram, export_png
                doc = generate_diagram(str(exp_dir))
                out_excalidraw = str(exp_dir / f"{exp_name_slug}.excalidraw")
                with open(out_excalidraw, "w") as f:
                    json.dump(doc, f, indent=2)
                out_png = str(exp_dir / f"{exp_name_slug}.png")
                if export_png(out_excalidraw, out_png):
                    st.success("Diagram regenerated")
                    diagram_png = Path(out_png)
                else:
                    st.warning("Excalidraw saved — PNG export failed (needs Node.js)")
                st.rerun()
            except Exception as e:
                st.error(f"Generation failed: {e}")

with diagram_col1:
    if diagram_png.exists():
        st.image(str(diagram_png), use_container_width=True)
    else:
        st.info("No diagram yet — click **Regenerate Diagram** to create one.")

# ── CatBoost training log (if exists) ──────────────────────────────────────

catboost_log = exp_dir / "catboost_info" / "learn_error.tsv"
if catboost_log.exists():
    st.markdown("---")
    st.subheader("CatBoost Training Curve")
    cb_df = pd.read_csv(catboost_log, sep="\t")
    if len(cb_df) > 0:
        fig5 = px.line(cb_df, x="iter", y=cb_df.columns[1], template="plotly_dark")
        fig5.update_layout(height=350, xaxis_title="Iteration", yaxis_title=cb_df.columns[1])
        st.plotly_chart(fig5, use_container_width=True)

# ── Results table (kept experiments only) ──────────────────────────────────

if results_df is not None and experiments_df is not None:
    st.markdown("---")
    st.subheader("Results (Kept Experiments)")
    st.caption("From `results.tsv` — curated list of kept/baseline experiments only.")

    res_df = results_df.reset_index(drop=True)
    res_df.insert(0, "#", range(1, len(res_df) + 1))

    res_metric_cols = [c for c in res_df.columns if is_metric_col(c, res_df)]
    if primary and primary in res_df.columns:
        def highlight_best_res(s):
            if s.name != primary:
                return [""] * len(s)
            numeric = pd.to_numeric(s, errors="coerce")
            best = numeric.min() if direction == "lower" else numeric.max()
            return ["background-color: rgba(63,185,80,0.2)" if v == best else "" for v in numeric]

        styled_res = res_df.style.apply(highlight_best_res).format(
            {c: "{:,.2f}" for c in res_metric_cols if c in res_df.columns},
            na_rep="—",
        )
        st.dataframe(styled_res, use_container_width=True, height=300)
    else:
        st.dataframe(res_df, use_container_width=True, height=300)

# ── Full experiment table ───────────────────────────────────────────────────

st.markdown("---")
st.subheader("Experiment Log")

if has_status:
    statuses = ["All"] + sorted(df["status"].dropna().unique().tolist())
    selected_status = st.selectbox("Filter by status", statuses)
    display_df = df if selected_status == "All" else df[df["status"] == selected_status]
else:
    display_df = df

# Highlight best value
if primary and primary in display_df.columns:
    def highlight_best(s):
        if s.name != primary:
            return [""] * len(s)
        numeric = pd.to_numeric(s, errors="coerce")
        best = numeric.min() if direction == "lower" else numeric.max()
        return ["background-color: rgba(63,185,80,0.2)" if v == best else "" for v in numeric]

    styled = display_df.style.apply(highlight_best).format(
        {c: "{:,.2f}" for c in metric_cols if c in display_df.columns},
        na_rep="—",
    )
    st.dataframe(styled, use_container_width=True, height=500)
else:
    st.dataframe(display_df, use_container_width=True, height=500)

# ── Experiment detail viewer ──────────────────────────────────────────────

if has_description and primary and len(df) > 1:
    st.markdown("---")
    st.subheader("Experiment Detail")

    exp_options = df.dropna(subset=[primary]).apply(
        lambda r: f"#{int(r['#'])} — {r.get('description', '')[:60]} ({primary}={r[primary]:.4f})", axis=1
    ).tolist()
    if exp_options:
        selected_exp = st.selectbox("Select experiment", exp_options)
        exp_num = int(selected_exp.split("#")[1].split(" ")[0])
        exp_row = df[df["#"] == exp_num].iloc[0]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Metrics**")
            detail_data = {c: f"{exp_row[c]:,.4f}" for c in metric_cols if pd.notna(exp_row.get(c))}
            st.json(detail_data)
        with col_b:
            st.markdown("**Info**")
            info_data = {}
            if has_status:
                info_data["Status"] = str(exp_row.get("status", "—"))
            if has_description:
                info_data["Description"] = str(exp_row.get("description", "—"))
            if "commit" in df.columns:
                info_data["Commit"] = str(exp_row.get("commit", "—"))
            st.json(info_data)

# ── Multi-experiment comparison ───────────────────────────────────────────

if len(experiments) > 1:
    st.markdown("---")
    st.subheader("Cross-Experiment Comparison")

    compare_names = st.multiselect("Compare experiments", exp_names, default=exp_names[:min(4, len(exp_names))])
    if compare_names:
        compare_data = []
        for name in compare_names:
            cdir = EXPERIMENTS_DIR / name
            c_results = load_tsv(cdir / "results.tsv")
            c_experiments = load_tsv(cdir / "experiments.tsv")
            c_df = c_results if c_results is not None else c_experiments
            c_meta = parse_program_md(cdir / "program.md")
            if c_df is not None:
                c_primary = c_meta.get("primary_metric", primary)
                if c_primary and c_primary in c_df.columns:
                    c_valid = pd.to_numeric(c_df[c_primary], errors="coerce").dropna()
                    c_dir = c_meta.get("direction") or metric_direction(c_primary)
                    c_best = c_valid.min() if c_dir == "lower" else c_valid.max()
                    c_baseline = c_valid.iloc[0] if len(c_valid) > 0 else None
                    compare_data.append({
                        "Experiment": name,
                        "Metric": c_primary,
                        "Baseline": f"{c_baseline:.4f}" if c_baseline is not None else "—",
                        "Best": f"{c_best:.4f}",
                        "Iterations": len(c_df),
                    })
        if compare_data:
            st.dataframe(pd.DataFrame(compare_data), use_container_width=True)

# ── Code & Infra ──────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Code & Infrastructure")

code_tab_train, code_tab_prepare, code_tab_infra = st.tabs(["train.py", "prepare.py", "Infrastructure"])

with code_tab_train:
    train_py = exp_dir / "train.py"
    if train_py.exists():
        # Show commit tag for current code
        has_commit_col = "commit" in df.columns
        if has_commit_col:
            latest_commit = df.iloc[-1].get("commit", "—")
            best_commit = best_row.get("commit", "—") if 'best_row' in dir() else "—"
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.markdown(f"**Latest commit:** `{latest_commit}`")
            with tc2:
                st.markdown(f"**Best experiment commit:** `{best_commit}`")
            with tc3:
                if meta.get("tag"):
                    st.markdown(f"**Experiment tag:** `{meta['tag']}`")
        elif meta.get("tag"):
            st.markdown(f"**Experiment tag:** `{meta['tag']}`")

        st.code(train_py.read_text(), language="python", line_numbers=True)
    else:
        st.info("No `train.py` found in this experiment.")

with code_tab_prepare:
    prepare_py = exp_dir / "prepare.py"
    if prepare_py.exists():
        st.code(prepare_py.read_text(), language="python", line_numbers=True)
    else:
        st.info("No `prepare.py` found in this experiment.")

with code_tab_infra:
    infra_col1, infra_col2 = st.columns(2)

    with infra_col1:
        st.markdown("**Experiment Config**")
        infra_data = {}
        if meta.get("tag"):
            infra_data["Tag"] = meta["tag"]
        if meta.get("backend"):
            infra_data["Backend"] = meta["backend"]
        if meta.get("agent_llm"):
            infra_data["Agent LLM"] = meta["agent_llm"]
        if meta.get("task"):
            infra_data["Task"] = meta["task"]
        if meta.get("target"):
            infra_data["Target"] = meta["target"]
        if primary:
            infra_data["Primary Metric"] = f"{primary} ({direction} is better)"
        infra_data["Total Experiments"] = len(df)
        if has_status:
            infra_data["Kept"] = int(df[df["status"].isin(["keep", "baseline"])].shape[0])
            infra_data["Reverted"] = int(df[df["status"] == "revert"].shape[0])
            infra_data["Crashed"] = int(df[df["status"] == "crash"].shape[0])
        st.json(infra_data)

    with infra_col2:
        st.markdown("**Dependencies**")
        pyproject = exp_dir / "pyproject.toml"
        if pyproject.exists():
            st.code(pyproject.read_text(), language="toml")
        else:
            st.info("No `pyproject.toml` found.")

    # Commit log from experiments.tsv
    if "commit" in df.columns:
        st.markdown("**Commit History**")
        commit_df = df[["#", "commit", "status", "description"]].copy() if has_description else df[["#", "commit", "status"]].copy()
        commit_df = commit_df.dropna(subset=["commit"])
        st.dataframe(commit_df, use_container_width=True, height=300)

    # Git log from experiment repo if available
    exp_git_dir = exp_dir / ".git"
    if exp_git_dir.exists():
        st.markdown("**Git Log (last 15 commits)**")
        try:
            git_log = subprocess.run(
                ["git", "log", "--oneline", "--no-decorate", "-15"],
                capture_output=True, text=True, cwd=str(exp_dir), timeout=5,
            )
            if git_log.returncode == 0 and git_log.stdout.strip():
                st.code(git_log.stdout.strip(), language="text")
        except Exception:
            pass
