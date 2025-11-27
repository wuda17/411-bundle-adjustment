#!/usr/bin/env python3
"""Parse solver log files (robust_logs.txt, sparse_logs.txt) and produce plots.

Usage:
  python plot_logs.py [--logs file1 file2 ...]

Outputs are saved to `outputs/` as PNG files.
"""
import argparse
import ast
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt


def ensure_outdir(path: str = "outputs"):
    os.makedirs(path, exist_ok=True)
    return path


def _parse_dict_line(line: str):
    # Safely parse a Python-style dict printed in the logs
    try:
        return ast.literal_eval(line.strip())
    except Exception:
        # attempt to clean common `np.float64(...)` wrappers
        cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", line)
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return None


def parse_log_file(path: str) -> Dict:
    """
    Parse the log file into a structure containing:
      - dataset: dict
      - experiments: list of {label, iterations: list of dicts}
      - solver_metrics: dict (from '=== Solver / Problem Metrics ===')
      - benchmark: dict (from '=== Benchmarking Metrics ===')
    """
    data = {
        "dataset": None,
        "experiments": [],
        "solver_metrics": None,
        "benchmark": None,
    }

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Dataset header
    m = re.search(r"=== Dataset Loaded ===\n(\{.*?\})", text, re.S)
    if m:
        data["dataset"] = _parse_dict_line(m.group(1))

    # Experiments: find each '=== Running Experiment' block
    for exp_m in re.finditer(
        r"=== Running Experiment .*?: (.*?) ===\n(.*?)(?:=== Solver / Problem Metrics ===|$)",
        text,
        re.S,
    ):
        label = exp_m.group(1).strip()
        block = exp_m.group(2)

        # collect iteration lines like: <spaces><num><spaces><num><spaces><float> ...
        it_lines = []
        for line in block.splitlines():
            if re.match(r"\s*\d+\s+\d+\s+", line):
                it_lines.append(line)

        iterations = []
        for L in it_lines:
            # collapse multiple spaces, split
            parts = re.split(r"\s+", L.strip())
            # Expect at least: iter, total_nfev, cost
            try:
                it = int(parts[0])
                total_nfev = int(parts[1])
                cost = float(parts[2])
            except Exception:
                continue

            # remaining optional fields
            cost_red = float(parts[3]) if len(parts) > 3 and parts[3] != "" else None
            step_norm = float(parts[4]) if len(parts) > 4 and parts[4] != "" else None
            optimality = float(parts[5]) if len(parts) > 5 and parts[5] != "" else None

            iterations.append(
                {
                    "iteration": it,
                    "total_nfev": total_nfev,
                    "cost": cost,
                    "cost_reduction": cost_red,
                    "step_norm": step_norm,
                    "optimality": optimality,
                }
            )

        # solver/benchmark after this block
        data["experiments"].append({"label": label, "iterations": iterations})

    # Solver / Problem Metrics
    m2 = re.search(r"=== Solver / Problem Metrics ===\n(\{.*?\})", text, re.S)
    if m2:
        data["solver_metrics"] = _parse_dict_line(m2.group(1))

    # Benchmarking Metrics
    m3 = re.search(r"=== Benchmarking Metrics ===\n(\{.*?\})", text, re.S)
    if m3:
        data["benchmark"] = _parse_dict_line(m3.group(1))

    return data


def plot_cost_vs_iteration(parsed: Dict, outpath: str, tag: str):
    if not parsed["experiments"]:
        print(f"No experiment iterations found in {tag}")
        return None

    plt.figure(figsize=(8, 5))
    for exp in parsed["experiments"]:
        its = [it["iteration"] for it in exp["iterations"]]
        costs = [it["cost"] for it in exp["iterations"]]
        if not its:
            continue
        plt.plot(its, costs, marker=".", label=exp.get("label", "exp"))

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (log scale)")
    plt.title(f"Cost vs Iteration — {tag}")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.legend()
    out_file = os.path.join(outpath, f"{tag}_cost_vs_iteration.png")
    plt.tight_layout()
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")
    return out_file


def plot_summary_comparison(parsed_list: List[Dict], labels: List[str], outpath: str):
    # Compare total_runtime and peak_memory and final cost
    runtimes = []
    memories = []
    final_costs = []
    mean_cam_errors = []
    mean_pt_errors = []
    median_cam_errors = []
    median_pt_errors = []

    for p in parsed_list:
        bench = p.get("benchmark") or {}
        solver = p.get("solver_metrics") or {}
        runtimes.append(float(bench.get("total_runtime", float("nan"))))
        memories.append(float(bench.get("peak_memory_mb", float("nan"))))
        final_costs.append(float(solver.get("final_cost", float("nan"))))
        mean_cam_errors.append(float(solver.get("mean_cam_error", float("nan"))))
        median_cam_errors.append(float(solver.get("median_cam_error", float("nan"))))
        mean_pt_errors.append(float(solver.get("mean_pt_error", float("nan"))))
        median_pt_errors.append(float(solver.get("median_pt_error", float("nan"))))

    x = range(len(labels))

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.bar(x, runtimes)
    plt.xticks(x, labels)
    plt.title("Total runtime (s)")

    plt.subplot(2, 2, 2)
    plt.bar(x, memories)
    plt.xticks(x, labels)
    plt.title("Peak memory (MB)")

    plt.subplot(2, 2, 3)
    plt.bar(x, final_costs)
    plt.xticks(x, labels)
    plt.title("Final cost")

    plt.subplot(2, 2, 4)
    # grouped bars: cam_mean, cam_median, pt_mean, pt_median per label
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    plt.bar([i + offsets[0] for i in x], mean_cam_errors, width=width, label="cam mean")
    plt.bar([i + offsets[1] for i in x], median_cam_errors, width=width, label="cam median")
    plt.bar([i + offsets[2] for i in x], mean_pt_errors, width=width, label="pt mean")
    plt.bar([i + offsets[3] for i in x], median_pt_errors, width=width, label="pt median")
    plt.xticks(x, labels)
    plt.title("Camera / Point Errors (mean & median)")
    plt.legend()

    plt.tight_layout()
    out_file = os.path.join(outpath, "summary_comparison.png")
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs",
        nargs="*",
        default=["robust_logs.txt", "sparse_logs.txt"],
        help="Paths to log files to parse",
    )
    parser.add_argument("--outdir", default="outputs", help="Directory to save plots")
    args = parser.parse_args()

    outdir = ensure_outdir(args.outdir)

    parsed_all = []
    labels = []
    for p in args.logs:
        if not os.path.exists(p):
            print(f"Log file not found: {p}")
            continue
        parsed = parse_log_file(p)
        label = os.path.splitext(os.path.basename(p))[0]
        labels.append(label)
        parsed_all.append(parsed)

    if parsed_all:
        plot_summary_comparison(parsed_all, labels, outdir)


if __name__ == "__main__":
    main()
