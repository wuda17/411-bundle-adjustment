import os
from typing import List, Tuple

import numpy as np
import psutil
import matplotlib.pyplot as plt
import pandas as pd


# -----------------------------
# 1. Logger classes (in-memory)
# -----------------------------
class SolverLogger:
    """Collects per-iteration solver metrics and per-camera/point aggregates.

    The logger keeps a list `history` of dicts. This class is intentionally
    lightweight: callers provide residuals and optionally Jacobian/step.
    """

    def __init__(self, n_cameras: int, n_points: int, camera_indices, point_indices):
        self.n_cameras = n_cameras
        self.n_points = n_points
        self.cam_idx = np.asarray(camera_indices)
        self.pt_idx = np.asarray(point_indices)
        self.history: List[dict] = []

    def log_iteration(
        self, iteration: int, residuals: np.ndarray, J=None, step=None, damping=None
    ):
        cost = float(0.5 * np.sum(residuals**2))
        residual_norm = float(np.linalg.norm(residuals))
        grad_norm = float(np.linalg.norm(J.T @ residuals)) if J is not None else None
        step_norm = float(np.linalg.norm(step)) if step is not None else None

        # per-observation reprojection error (observations are 2D residuals)
        per_obs = np.linalg.norm(residuals.reshape(-1, 2), axis=1)

        def agg_per_index(idx_array, n):
            if len(idx_array) == 0:
                return [], float(np.nan), float(np.nan), float(np.nan)
            arr = []
            for i in range(n):
                mask = idx_array == i
                if np.any(mask):
                    arr.append(float(np.mean(per_obs[mask])))
                else:
                    arr.append(float(np.nan))
            return (
                arr,
                float(np.nanmean(arr)),
                float(np.nanmedian(arr)),
                float(np.nanmax(arr)),
            )

        per_cam_error, mean_cam, median_cam, max_cam = agg_per_index(
            self.cam_idx, self.n_cameras
        )
        per_pt_error, mean_pt, median_pt, max_pt = agg_per_index(
            self.pt_idx, self.n_points
        )

        self.history.append(
            {
                "iteration": int(iteration),
                "cost": cost,
                "residual_norm": residual_norm,
                "grad_norm": grad_norm,
                "step_norm": step_norm,
                "damping": damping,
                "per_cam_error": per_cam_error,
                "per_pt_error": per_pt_error,
                "mean_cam_error": mean_cam,
                "median_cam_error": median_cam,
                "max_cam_error": max_cam,
                "mean_pt_error": mean_pt,
                "median_pt_error": median_pt,
                "max_pt_error": max_pt,
            }
        )

    def summary(self) -> dict:
        if not self.history:
            return {}
        final = self.history[-1]
        return {
            "final_cost": final["cost"],
            "residual_norm": final["residual_norm"],
            "grad_norm": final.get("grad_norm"),
            "step_norm": final.get("step_norm"),
            "mean_cam_error": final.get("mean_cam_error"),
            "median_cam_error": final.get("median_cam_error"),
            "max_cam_error": final.get("max_cam_error"),
            "mean_pt_error": final.get("mean_pt_error"),
            "median_pt_error": final.get("median_pt_error"),
            "max_pt_error": final.get("max_pt_error"),
        }


class BenchmarkLogger:
    """Collects per-iteration benchmarking metrics (time, jac time, memory)."""

    def __init__(self):
        self.history: List[dict] = []
        self.total_runtime = 0.0
        self.total_jac_time = 0.0

    def log_iteration(
        self, iteration: int, iteration_time: float, jac_time: float = None
    ):
        mem_mb = psutil.Process().memory_info().rss / 1e6
        self.total_runtime += float(iteration_time)
        if jac_time:
            self.total_jac_time += float(jac_time)
        self.history.append(
            {
                "iteration": int(iteration),
                "iteration_time": float(iteration_time),
                "jacobian_time": jac_time,
                "memory_mb": float(mem_mb),
            }
        )

    def summary(self) -> dict:
        peak = float(max((h["memory_mb"] for h in self.history), default=float(np.nan)))
        return {
            "total_runtime": self.total_runtime,
            "total_jacobian_time": self.total_jac_time,
            "peak_memory_mb": peak,
        }


# -----------------------------
# 2. Pandas-backed I/O and plotting helpers
# -----------------------------
def log_iteration_metrics(
    solver: SolverLogger,
    bench: BenchmarkLogger,
    experiment_number: int = 1,
    label: str = "Experiment",
    output_dir: str = "outputs",
) -> Tuple[str, str]:
    """Write per-iteration CSVs for solver and benchmark using pandas.

    Returns (solver_csv_path, bench_csv_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    solver_rows = []
    for h in solver.history:
        solver_rows.append(
            {
                "iteration": h.get("iteration"),
                "cost": h.get("cost"),
                "residual_norm": h.get("residual_norm"),
                "grad_norm": h.get("grad_norm"),
                "step_norm": h.get("step_norm"),
                "damping": h.get("damping"),
                "mean_cam_error": h.get("mean_cam_error"),
                "median_cam_error": h.get("median_cam_error"),
                "max_cam_error": h.get("max_cam_error"),
                "mean_pt_error": h.get("mean_pt_error"),
                "median_pt_error": h.get("median_pt_error"),
                "max_pt_error": h.get("max_pt_error"),
            }
        )
    solver_df = pd.DataFrame(solver_rows)
    solver_fname = os.path.join(
        output_dir, f"exp{experiment_number}_{label}_solver_iterations.csv"
    )
    solver_df.to_csv(solver_fname, index=False)

    bench_rows = [
        {
            "iteration": h.get("iteration"),
            "iteration_time": h.get("iteration_time"),
            "jacobian_time": h.get("jacobian_time"),
            "memory_mb": h.get("memory_mb"),
        }
        for h in bench.history
    ]
    bench_df = pd.DataFrame(bench_rows)
    bench_fname = os.path.join(
        output_dir, f"exp{experiment_number}_{label}_bench_iterations.csv"
    )
    bench_df.to_csv(bench_fname, index=False)

    return solver_fname, bench_fname


def plot_iteration_metrics_from_files(
    solver_csv: str, bench_csv: str, out_png: str = None, title: str = None
):
    """Read per-iteration CSVs and plot diagnostics. Returns matplotlib Figure."""
    solver_df = pd.read_csv(solver_csv)
    bench_df = pd.read_csv(bench_csv)

    it = solver_df["iteration"].values
    costs = solver_df["cost"].values
    residuals = solver_df["residual_norm"].values
    grad_norms = (
        solver_df["grad_norm"].values
        if "grad_norm" in solver_df.columns
        else np.full_like(it, np.nan, dtype=float)
    )
    step_norms = (
        solver_df["step_norm"].values
        if "step_norm" in solver_df.columns
        else np.full_like(it, np.nan, dtype=float)
    )

    iter_times = (
        bench_df["iteration_time"].values
        if "iteration_time" in bench_df.columns
        else np.array([])
    )
    mems = (
        bench_df["memory_mb"].values
        if "memory_mb" in bench_df.columns
        else np.array([])
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].plot(it, costs, marker=".")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Cost vs Iteration (from CSV)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(it, residuals, marker=".")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Residual Norm vs Iteration (from CSV)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(it, step_norms, marker=".")
    axes[0, 2].set_title("Step Norm vs Iteration (from CSV)")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(np.arange(len(iter_times)), iter_times, marker=".")
    axes[1, 0].set_title("Iteration Time (s) (from CSV)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(np.arange(len(mems)), mems, marker=".")
    axes[1, 1].set_title("Memory Usage (MB) (from CSV)")
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(it, grad_norms, marker=".")
    axes[1, 2].set_title("Gradient Norm (from CSV)")
    axes[1, 2].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
        plt.close(fig)
        print(f"Saved iteration plot: {out_png}")
    return fig


def plot_iteration_metrics(
    solver: SolverLogger,
    bench: BenchmarkLogger,
    experiment_number: int = 1,
    label: str = "Experiment",
    output_dir: str = "outputs",
):
    """Compatibility wrapper: write CSVs then plot from them."""
    solver_csv, bench_csv = log_iteration_metrics(
        solver, bench, experiment_number, label, output_dir
    )
    out_png = os.path.join(output_dir, f"iteration_metrics_exp{experiment_number}.png")
    return plot_iteration_metrics_from_files(
        solver_csv, bench_csv, out_png, title=f"Experiment {experiment_number}: {label}"
    )


def log_summary_metrics(
    solver_loggers: List[SolverLogger],
    bench_loggers: List[BenchmarkLogger],
    labels: List[str],
    output_dir: str = "outputs",
) -> str:
    """Write per-experiment iteration CSVs and a summary CSV for multiple experiments.

    Returns the path to the summary CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for idx, (slg, blg, label) in enumerate(
        zip(solver_loggers, bench_loggers, labels), start=1
    ):
        # write per-experiment CSVs
        solver_csv = os.path.join(output_dir, f"exp{idx}_{label}_solver_iterations.csv")
        pd.DataFrame(
            [
                {
                    "iteration": h.get("iteration"),
                    "cost": h.get("cost"),
                    "residual_norm": h.get("residual_norm"),
                    "grad_norm": h.get("grad_norm"),
                    "step_norm": h.get("step_norm"),
                    "damping": h.get("damping"),
                    "mean_cam_error": h.get("mean_cam_error"),
                    "median_cam_error": h.get("median_cam_error"),
                    "max_cam_error": h.get("max_cam_error"),
                    "mean_pt_error": h.get("mean_pt_error"),
                    "median_pt_error": h.get("median_pt_error"),
                    "max_pt_error": h.get("max_pt_error"),
                }
                for h in slg.history
            ]
        ).to_csv(solver_csv, index=False)

        bench_csv = os.path.join(output_dir, f"exp{idx}_{label}_bench_iterations.csv")
        pd.DataFrame(
            [
                {
                    "iteration": h.get("iteration"),
                    "iteration_time": h.get("iteration_time"),
                    "jacobian_time": h.get("jacobian_time"),
                    "memory_mb": h.get("memory_mb"),
                }
                for h in blg.history
            ]
        ).to_csv(bench_csv, index=False)

        s = slg.summary()
        b = blg.summary()
        rows.append(
            {
                "label": label,
                "final_cost": s.get("final_cost"),
                "residual_norm": s.get("residual_norm"),
                "grad_norm": s.get("grad_norm"),
                "step_norm": s.get("step_norm"),
                "mean_cam_error": s.get("mean_cam_error"),
                "median_cam_error": s.get("median_cam_error"),
                "max_cam_error": s.get("max_cam_error"),
                "mean_pt_error": s.get("mean_pt_error"),
                "median_pt_error": s.get("median_pt_error"),
                "max_pt_error": s.get("max_pt_error"),
                "total_runtime": b.get("total_runtime"),
                "total_jacobian_time": b.get("total_jacobian_time"),
                "peak_memory_mb": b.get("peak_memory_mb"),
            }
        )

    summary_csv = os.path.join(output_dir, "summary_comparison.csv")
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(f"Wrote summary CSV: {summary_csv}")
    return summary_csv


def plot_summary_metrics_from_file(summary_csv: str, out_png: str = None):
    df = pd.read_csv(summary_csv)
    labels = df["label"].astype(str).tolist()
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].bar(x, df["final_cost"].values)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0, 0].set_title("Final Cost")

    axes[0, 1].bar(x, df["total_runtime"].values)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=45, ha="right")
    axes[0, 1].set_title("Total Runtime (s)")

    axes[1, 0].bar(x, df["peak_memory_mb"].values)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=45, ha="right")
    axes[1, 0].set_title("Peak Memory (MB)")

    width = 0.35
    axes[1, 1].bar(
        x - width / 2, df["mean_cam_error"].values, width=width, label="cam mean"
    )
    axes[1, 1].bar(
        x + width / 2, df["median_cam_error"].values, width=width, label="cam median"
    )
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1, 1].set_title("Camera Errors")
    axes[1, 1].legend()

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png)
        plt.close(fig)
        print(f"Saved summary plot: {out_png}")
    return fig


def plot_summary_comparison(
    solver_loggers: List[SolverLogger],
    bench_loggers: List[BenchmarkLogger],
    labels: List[str],
    output_dir: str = "outputs",
):
    summary_csv = log_summary_metrics(solver_loggers, bench_loggers, labels, output_dir)
    out_png = os.path.join(output_dir, "summary_comparison.png")
    return plot_summary_metrics_from_file(summary_csv, out_png)

    # New functions to discover iteration pairs and main CLI

def _discover_iteration_pairs(output_dir: str = "outputs"):
    """Return list of (solver_csv, bench_csv) pairs found under `output_dir`."""
    if not os.path.isdir(output_dir):
        return []
    files = sorted(os.listdir(output_dir))
    solver_files = [f for f in files if f.endswith("_solver_iterations.csv")]
    pairs = []
    for sf in solver_files:
        prefix = sf[: -len("_solver_iterations.csv")]
        bf = prefix + "_bench_iterations.csv"
        sfp = os.path.join(output_dir, sf)
        bfp = os.path.join(output_dir, bf)
        if os.path.exists(bfp):
            pairs.append((sfp, bfp))
    return pairs


def main():
    """CLI to scan `outputs/`, plot per-experiment iteration diagnostics and a summary.

    Behavior:
    - For each `exp*_solver_iterations.csv`/`exp*_bench_iterations.csv` pair, create
      `exp*_iterations.png` using `plot_iteration_metrics_from_files`.
    - If `summary_comparison.csv` exists, plot it. Otherwise aggregate per-pair
      summary rows, write `summary_comparison.csv` and plot it.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="metrics_logger")
    parser.add_argument("--output-dir", default="outputs", help="Directory with CSV outputs")
    args = parser.parse_args()

    outdir = args.output_dir
    pairs = _discover_iteration_pairs(outdir)
    if not pairs:
        print(f"No iteration CSV pairs found under: {outdir}")
    for solver_csv, bench_csv in pairs:
        try:
            base = os.path.splitext(os.path.basename(solver_csv))[0]
            out_png = os.path.join(outdir, base + "_iterations.png")
            print(f"Plotting iterations for {base} -> {out_png}")
            plot_iteration_metrics_from_files(solver_csv, bench_csv, out_png, title=base)
        except Exception as e:
            print(f"Failed to plot {solver_csv} / {bench_csv}: {e}")

    summary_csv = os.path.join(outdir, "summary_comparison.csv")
    if os.path.exists(summary_csv):
        print(f"Found existing summary CSV: {summary_csv} — plotting")
        plot_summary_metrics_from_file(summary_csv, os.path.join(outdir, "summary_comparison.png"))
        return

    # Build a summary from discovered pairs
    rows = []
    for solver_csv, bench_csv in pairs:
        try:
            sdf = pd.read_csv(solver_csv)
            bdf = pd.read_csv(bench_csv)
            if sdf.empty:
                continue
            last = sdf.iloc[-1]
            label = os.path.splitext(os.path.basename(solver_csv))[0]
            rows.append(
                {
                    "label": label,
                    "final_cost": float(last.get("cost", float("nan"))),
                    "residual_norm": float(last.get("residual_norm", float("nan"))),
                    "mean_cam_error": float(last.get("mean_cam_error", float("nan"))),
                    "median_cam_error": float(last.get("median_cam_error", float("nan"))),
                    "mean_pt_error": float(last.get("mean_pt_error", float("nan"))),
                    "median_pt_error": float(last.get("median_pt_error", float("nan"))),
                    "total_runtime": float(bdf["iteration_time"].sum()) if "iteration_time" in bdf.columns else float("nan"),
                    "peak_memory_mb": float(bdf["memory_mb"].max()) if "memory_mb" in bdf.columns else float("nan"),
                }
            )
        except Exception as e:
            print(f"Failed to aggregate {solver_csv} / {bench_csv}: {e}")

    if rows:
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(summary_csv, index=False)
        print(f"Wrote summary CSV: {summary_csv}")
        plot_summary_metrics_from_file(summary_csv, os.path.join(outdir, "summary_comparison.png"))
    else:
        print("No summary rows produced — nothing to plot.")


if __name__ == "__main__":
    main()
