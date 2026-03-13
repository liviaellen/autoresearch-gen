#!/usr/bin/env python3
"""
Generate an Excalidraw diagram for a specific experiment directory.

Reads the experiment's train.py, results.tsv, and experiments.tsv to build
a visual architecture diagram showing model config, training flow, and results.

Usage:
    python excalidraw_gen.py experiments/my-run
"""

import argparse
import json
import os
import re
import subprocess
import sys


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_train_py(path):
    """Extract model config from train.py."""
    config = {}
    if not os.path.isfile(path):
        return config
    with open(path) as f:
        src = f.read()

    patterns = {
        "n_layers": r"['\"]?n_layers?['\"]?\s*[:=]\s*(\d+)",
        "d_model": r"['\"]?d_model['\"]?\s*[:=]\s*(\d+)",
        "n_heads": r"['\"]?n_heads?['\"]?\s*[:=]\s*(\d+)",
        "n_kv_heads": r"['\"]?n_kv_heads?['\"]?\s*[:=]\s*(\d+)",
        "vocab_size": r"['\"]?vocab_size['\"]?\s*[:=]\s*(\d+)",
        "seq_len": r"(?:MAX_SEQ_LEN|seq_len|context_length)\s*=\s*(\d+)",
        "batch_size": r"(?:BATCH_SIZE|batch_size)\s*=\s*(\d+)",
        "learning_rate": r"(?:LEARNING_RATE|learning_rate|lr)\s*=\s*([0-9.e-]+)",
        "time_budget": r"TIME_BUDGET\s*=\s*(\d+)",
        "num_params_M": r"num_params.*?/\s*1e6",
    }
    for key, pat in patterns.items():
        m = re.search(pat, src, re.IGNORECASE)
        if m:
            config[key] = m.group(1) if m.lastindex else True

    # Detect architecture features
    if re.search(r"GQA|grouped.query|n_kv_heads", src, re.IGNORECASE):
        config["attention"] = "GQA"
    elif re.search(r"MHA|multi.head.attention", src, re.IGNORECASE):
        config["attention"] = "MHA"
    elif re.search(r"sliding.window", src, re.IGNORECASE):
        config["attention"] = "Sliding Window"
    else:
        config["attention"] = "Standard"

    if re.search(r"RMSNorm|rms_norm", src):
        config["norm"] = "RMSNorm"
    elif re.search(r"LayerNorm|layer_norm", src):
        config["norm"] = "LayerNorm"

    if re.search(r"SwiGLU|swiglu", src, re.IGNORECASE):
        config["ffn"] = "SwiGLU"
    elif re.search(r"GeGLU|geglu", src, re.IGNORECASE):
        config["ffn"] = "GeGLU"
    elif re.search(r"GELU|gelu", src):
        config["ffn"] = "GELU"
    elif re.search(r"ReLU|relu", src):
        config["ffn"] = "ReLU"

    # Detect backend
    if re.search(r"import mlx", src):
        config["backend"] = "MLX"
    elif re.search(r"import torch", src):
        config["backend"] = "PyTorch"

    return config


def parse_results_tsv(path):
    """Parse results.tsv and return list of dicts."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path) as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split("\t")
            if header is None:
                header = cols
                continue
            row = dict(zip(header, cols))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Excalidraw element builders
# ---------------------------------------------------------------------------

_SEED = 1000


def _next_seed():
    global _SEED
    _SEED += 1
    return _SEED


def _base(etype, eid, x, y, w, h, **overrides):
    el = {
        "type": etype,
        "id": eid,
        "x": x, "y": y,
        "width": w, "height": h,
        "angle": 0,
        "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 0,
        "opacity": 100,
        "seed": _next_seed(),
        "version": 1,
        "versionNonce": _next_seed(),
        "isDeleted": False,
        "groupIds": [],
        "boundElements": None,
        "link": None,
        "locked": False,
    }
    el.update(overrides)
    return el


def make_rect(eid, x, y, w, h, bg="transparent", stroke="#1e1e1e", radius=8):
    el = _base("rectangle", eid, x, y, w, h,
               backgroundColor=bg, strokeColor=stroke, roundness={"type": 3, "value": radius})
    return el


def make_text(eid, x, y, text, size=16, color="#1e1e1e", align="left", w=None, h=None, family=3):
    lines = text.split("\n")
    est_w = w or max(len(l) for l in lines) * size * 0.6
    est_h = h or len(lines) * size * 1.4
    el = _base("text", eid, x, y, est_w, est_h,
               strokeColor=color, backgroundColor="transparent",
               text=text, originalText=text,
               fontSize=size, fontFamily=family,
               textAlign=align, verticalAlign="top",
               containerId=None, lineHeight=1.25)
    return el


def make_arrow(eid, x1, y1, x2, y2, color="#1e1e1e"):
    el = _base("arrow", eid, x1, y1, abs(x2 - x1) or 1, abs(y2 - y1) or 1,
               strokeColor=color,
               points=[[0, 0], [x2 - x1, y2 - y1]],
               startBinding=None, endBinding=None,
               startArrowhead=None, endArrowhead="arrow",
               roundness={"type": 2})
    return el


# ---------------------------------------------------------------------------
# Diagram generator
# ---------------------------------------------------------------------------

def generate_diagram(exp_dir):
    """Build an Excalidraw JSON for the given experiment directory."""
    train_path = os.path.join(exp_dir, "train.py")
    results_path = os.path.join(exp_dir, "results.tsv")
    experiments_path = os.path.join(exp_dir, "experiments.tsv")

    config = parse_train_py(train_path)
    results = parse_results_tsv(results_path)
    all_experiments = parse_results_tsv(experiments_path)

    exp_name = os.path.basename(os.path.normpath(exp_dir))
    elements = []

    # --- Title ---
    elements.append(make_text("title", 250, 20, exp_name,
                              size=28, color="#1e40af", align="center", w=400))

    # --- Model Architecture box ---
    box_x, box_y = 40, 80
    box_w, box_h = 280, 260
    elements.append(make_rect("arch_box", box_x, box_y, box_w, box_h,
                              bg="#dbeafe", stroke="#3b82f6"))
    elements.append(make_text("arch_title", box_x + 15, box_y + 12,
                              "Model Architecture", size=18, color="#1e40af"))

    arch_lines = []
    if config.get("backend"):
        arch_lines.append(f"Backend:    {config['backend']}")
    if config.get("n_layers"):
        arch_lines.append(f"Layers:     {config['n_layers']}")
    if config.get("d_model"):
        arch_lines.append(f"d_model:    {config['d_model']}")
    if config.get("n_heads"):
        arch_lines.append(f"Heads:      {config['n_heads']}")
    if config.get("n_kv_heads"):
        arch_lines.append(f"KV Heads:   {config['n_kv_heads']}")
    if config.get("attention"):
        arch_lines.append(f"Attention:  {config['attention']}")
    if config.get("norm"):
        arch_lines.append(f"Norm:       {config['norm']}")
    if config.get("ffn"):
        arch_lines.append(f"FFN:        {config['ffn']}")
    if config.get("vocab_size"):
        arch_lines.append(f"Vocab:      {config['vocab_size']}")
    if config.get("seq_len"):
        arch_lines.append(f"Seq len:    {config['seq_len']}")

    if not arch_lines:
        arch_lines.append("(no train.py found)")

    elements.append(make_text("arch_detail", box_x + 15, box_y + 42,
                              "\n".join(arch_lines), size=14, color="#1e3a5f"))

    # --- Training Config box ---
    tc_x, tc_y = 370, 80
    tc_w, tc_h = 250, 160
    elements.append(make_rect("train_box", tc_x, tc_y, tc_w, tc_h,
                              bg="#fef3c7", stroke="#f59e0b"))
    elements.append(make_text("train_title", tc_x + 15, tc_y + 12,
                              "Training Config", size=18, color="#92400e"))

    train_lines = []
    if config.get("batch_size"):
        train_lines.append(f"Batch size:  {config['batch_size']}")
    if config.get("learning_rate"):
        train_lines.append(f"LR:          {config['learning_rate']}")
    if config.get("time_budget"):
        budget_s = int(config["time_budget"])
        train_lines.append(f"Time budget: {budget_s}s ({budget_s // 60}m)")

    if not train_lines:
        train_lines.append("(defaults)")

    elements.append(make_text("train_detail", tc_x + 15, tc_y + 42,
                              "\n".join(train_lines), size=14, color="#78350f"))

    # --- Results box ---
    res_x, res_y = 370, 270
    res_w, res_h = 250, 200
    elements.append(make_rect("results_box", res_x, res_y, res_w, res_h,
                              bg="#dcfce7", stroke="#22c55e"))
    elements.append(make_text("results_title", res_x + 15, res_y + 12,
                              "Results", size=18, color="#166534"))

    if results:
        # Find baseline and best
        baseline = results[0] if results else {}
        best = baseline
        metric_key = None
        for k in ["val_bpb", "cv_rmse", "accuracy", "f1"]:
            if k in baseline:
                metric_key = k
                break

        if metric_key:
            lower_better = metric_key in ("val_bpb", "cv_rmse")
            for r in results:
                try:
                    val = float(r.get(metric_key, "inf"))
                    best_val = float(best.get(metric_key, "inf"))
                    if lower_better and val < best_val:
                        best = r
                    elif not lower_better and val > best_val:
                        best = r
                except ValueError:
                    pass

            res_lines = [
                f"Metric:     {metric_key}",
                f"Baseline:   {baseline.get(metric_key, '?')}",
                f"Best:       {best.get(metric_key, '?')}",
                f"Experiments: {len(results)}",
            ]
            if best.get("num_params_M"):
                res_lines.append(f"Best params: {best['num_params_M']}M")
        else:
            res_lines = [f"Experiments: {len(results)}"]
    elif all_experiments:
        res_lines = [
            f"Iterations:  {len(all_experiments)}",
            "(no keeps yet)",
        ]
    else:
        res_lines = ["(no results yet — run baseline first)"]

    elements.append(make_text("results_detail", res_x + 15, res_y + 42,
                              "\n".join(res_lines), size=14, color="#14532d"))

    # --- Flow: pipeline boxes ---
    pipe_y = 510
    steps = [
        ("prepare", "prepare.py", "#e0e7ff", "#6366f1", "Download data\nTrain tokenizer"),
        ("train", "train.py", "#fce7f3", "#ec4899", "Model + loop\n(agent modifies)"),
        ("eval", "Evaluate", "#f0fdf4", "#22c55e", "Parse val_bpb\nLog to TSV"),
    ]

    step_w, step_h = 160, 90
    gap = 40
    total_w = len(steps) * step_w + (len(steps) - 1) * gap
    start_x = (res_x + res_w + box_x) // 2 - total_w // 2

    for i, (sid, label, bg, stroke, detail) in enumerate(steps):
        sx = start_x + i * (step_w + gap)
        elements.append(make_rect(f"pipe_{sid}", sx, pipe_y, step_w, step_h,
                                  bg=bg, stroke=stroke))
        elements.append(make_text(f"pipe_{sid}_label", sx + 10, pipe_y + 8,
                                  label, size=16, color=stroke))
        elements.append(make_text(f"pipe_{sid}_detail", sx + 10, pipe_y + 32,
                                  detail, size=12, color="#555"))

        # Arrow to next step
        if i < len(steps) - 1:
            ax = sx + step_w
            ay = pipe_y + step_h // 2
            elements.append(make_arrow(f"arrow_{i}", ax + 2, ay, ax + gap - 2, ay,
                                       color="#888"))

    # --- Arrow from arch box to pipeline ---
    elements.append(make_arrow("arrow_arch_pipe",
                               box_x + box_w // 2, box_y + box_h,
                               start_x + step_w // 2, pipe_y, color="#3b82f6"))

    # --- Arrow from train config to pipeline ---
    elements.append(make_arrow("arrow_train_pipe",
                               tc_x + tc_w // 2, tc_y + tc_h,
                               start_x + step_w + gap + step_w // 2, pipe_y,
                               color="#f59e0b"))

    # --- Arrow from pipeline to results ---
    last_sx = start_x + (len(steps) - 1) * (step_w + gap)
    elements.append(make_arrow("arrow_pipe_results",
                               last_sx + step_w // 2, pipe_y,
                               res_x + res_w // 2, res_y + res_h, color="#22c55e"))

    # --- Agent loop annotation ---
    loop_y = pipe_y + step_h + 30
    elements.append(make_text("loop_label",
                              start_x + step_w + gap, loop_y,
                              "← agent loop: modify train.py → run → log →",
                              size=13, color="#888", align="center"))

    doc = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://github.com/liviaellen/autoresearch-gen",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": "#ffffff",
        },
        "files": {},
    }
    return doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def export_png(excalidraw_path, png_path):
    """Export an Excalidraw file to PNG using @vraksha/excalidraw-cli."""
    abs_path = os.path.abspath(excalidraw_path)
    work_dir = os.path.dirname(abs_path)
    filename = os.path.basename(abs_path)
    try:
        subprocess.run(
            ["npx", "-y", "@vraksha/excalidraw-cli", filename, "."],
            capture_output=True, text=True, timeout=120, check=True,
            cwd=work_dir,
        )
        expected = os.path.join(work_dir, os.path.splitext(filename)[0] + ".png")
        if os.path.isfile(expected) and os.path.abspath(expected) != os.path.abspath(png_path):
            os.rename(expected, png_path)
        return os.path.isfile(png_path)
    except FileNotFoundError:
        print("  npx not found — install Node.js to enable PNG export", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"  PNG export failed: {e.stderr.strip()}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print("  PNG export timed out", file=sys.stderr)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate an Excalidraw diagram for an experiment",
    )
    parser.add_argument("exp_dir", help="Experiment directory (e.g. experiments/my-run)")
    parser.add_argument("--no-png", action="store_true",
                        help="Skip PNG export")
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        print(f"Error: {args.exp_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    exp_name = os.path.basename(os.path.normpath(args.exp_dir))
    output = os.path.join(args.exp_dir, f"{exp_name}.excalidraw")
    doc = generate_diagram(args.exp_dir)

    with open(output, "w") as f:
        json.dump(doc, f, indent=2)

    n_elements = len(doc["elements"])
    print(f"Generated {output} ({n_elements} elements)")

    # Export PNG with the same base name
    if not args.no_png:
        png_path = os.path.splitext(output)[0] + ".png"
        if export_png(output, png_path):
            print(f"Exported {png_path}")
        else:
            print(f"To export manually: open {os.path.basename(output)} at https://excalidraw.com and export as PNG")
    else:
        print(f"Open at https://excalidraw.com — File → Open → select {os.path.basename(output)}")


if __name__ == "__main__":
    main()
