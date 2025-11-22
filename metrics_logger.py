import os

import numpy as np
import psutil
import matplotlib.pyplot as plt


# -----------------------------
# 1. Solver / Problem Metrics Logger
# -----------------------------
class SolverLogger:
    """
    Logs per-iteration solver and problem metrics:
    - Cost, residual norm, gradient norm, step norm
    - LM damping placeholder
    - Per-camera and per-point reprojection errors
    """

    def __init__(self, n_cameras, n_points, camera_indices, point_indices):
        self.n_cameras = n_cameras
        self.n_points = n_points
        self.cam_idx = camera_indices
        self.pt_idx = point_indices
        self.history = []

    def log_iteration(
        self, iteration, params, residuals, J=None, step=None, damping=None
    ):
        cost = 0.5 * np.sum(residuals**2)
        residual_norm = np.linalg.norm(residuals)
        grad_norm = np.linalg.norm(J.T @ residuals) if J is not None else None
        step_norm = np.linalg.norm(step) if step is not None else None

        # Per-observation error
        per_obs_error = np.linalg.norm(residuals.reshape(-1, 2), axis=1)
        per_cam_error = [
            np.mean(per_obs_error[self.cam_idx == i]) for i in range(self.n_cameras)
        ]
        per_pt_error = [
            np.mean(per_obs_error[self.pt_idx == i]) for i in range(self.n_points)
        ]

        self.history.append(
            {
                "iteration": iteration,
                "cost": cost,
                "residual_norm": residual_norm,
                "grad_norm": grad_norm,
                "step_norm": step_norm,
                "damping": damping,
                "per_cam_error": per_cam_error,
                "per_pt_error": per_pt_error,
            }
        )

    def summary(self):
        final = self.history[-1]
        return {
            "final_cost": final["cost"],
            "residual_norm": final["residual_norm"],
            "grad_norm": final["grad_norm"],
            "step_norm": final["step_norm"],
            "mean_cam_error": np.mean(final["per_cam_error"]),
            "median_cam_error": np.median(final["per_cam_error"]),
            "max_cam_error": np.max(final["per_cam_error"]),
            "mean_pt_error": np.mean(final["per_pt_error"]),
            "median_pt_error": np.median(final["per_pt_error"]),
            "max_pt_error": np.max(final["per_pt_error"]),
        }


# -----------------------------
# 2. Benchmarking / Performance Logger
# -----------------------------
class BenchmarkLogger:
    """
    Logs per-iteration benchmarking metrics:
    - Iteration runtime
    - Jacobian computation time (if measured)
    - Memory usage (RSS)
    """

    def __init__(self):
        self.history = []
        self.total_runtime = 0
        self.total_jac_time = 0

    def log_iteration(self, iteration, iteration_time, jac_time=None):
        mem_usage = psutil.Process().memory_info().rss / 1e6  # MB
        self.total_runtime += iteration_time
        self.total_jac_time += jac_time if jac_time else 0

        self.history.append(
            {
                "iteration": iteration,
                "iteration_time": iteration_time,
                "jacobian_time": jac_time,
                "memory_mb": mem_usage,
            }
        )

    def summary(self):
        peak_memory = max(h["memory_mb"] for h in self.history)
        return {
            "total_runtime": self.total_runtime,
            "total_jacobian_time": self.total_jac_time,
            "peak_memory_mb": peak_memory,
        }


# -----------------------------
# Per-iteration plotting utility
# -----------------------------
def plot_iteration_metrics(
    logger, bench_logger, experiment_number=1, label="Experiment", output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    iters = [h["iteration"] for h in logger.history]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0, 0].plot(iters, [h["cost"] for h in logger.history], label=label)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Cost vs Iteration")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iters, [h["residual_norm"] for h in logger.history], label=label)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Residual Norm vs Iteration")
    axes[0, 1].grid(True, alpha=0.3)

    step_norms = [h.get("step_norm", np.nan) for h in logger.history]
    axes[0, 2].plot(iters, step_norms, label=label)
    axes[0, 2].set_title("Step Norm vs Iteration")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(
        iters, [h["iteration_time"] for h in bench_logger.history], label=label
    )
    axes[1, 0].set_title("Iteration Time (s)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iters, [h["memory_mb"] for h in bench_logger.history], label=label)
    axes[1, 1].set_title("Memory Usage (MB)")
    axes[1, 1].grid(True, alpha=0.3)

    grad_norms = [h.get("grad_norm", np.nan) for h in logger.history]
    axes[1, 2].plot(iters, grad_norms, label=label)
    axes[1, 2].set_title("Gradient Norm")
    axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle(f"Experiment {experiment_number}: {label}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save figure
    fig_path = os.path.join(output_dir, f"iteration_metrics_exp{experiment_number}.png")
    plt.savefig(fig_path)
    plt.close(fig)


# -----------------------------
# Summary comparison plotting utility
# -----------------------------
def plot_summary_comparison(
    solver_loggers, bench_loggers, labels, output_dir="outputs"
):
    os.makedirs(output_dir, exist_ok=True)

    final_costs = [lg.summary()["final_cost"] for lg in solver_loggers]
    mean_cam_errors = [lg.summary()["mean_cam_error"] for lg in solver_loggers]
    mean_pt_errors = [lg.summary()["mean_pt_error"] for lg in solver_loggers]
    total_runtimes = [bg.summary()["total_runtime"] for bg in bench_loggers]
    peak_mem = [bg.summary()["peak_memory_mb"] for bg in bench_loggers]

    x = np.arange(len(labels))
    width = 0.35

    # Helper function to save bar charts
    def save_bar_chart(values, ylabel, title, fname):
        plt.figure(figsize=(12, 4))
        plt.bar(x, values, width, color="skyblue")
        plt.xticks(x, labels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

    save_bar_chart(
        final_costs, "Final Cost", "Final Cost Comparison", "summary_final_cost.png"
    )
    save_bar_chart(
        mean_cam_errors,
        "Mean Camera Error (pixels)",
        "Mean Camera Error Comparison",
        "summary_mean_cam_error.png",
    )
    save_bar_chart(
        mean_pt_errors,
        "Mean Point Error (pixels)",
        "Mean Point Error Comparison",
        "summary_mean_pt_error.png",
    )
    save_bar_chart(
        total_runtimes,
        "Total Runtime (s)",
        "Total Runtime Comparison",
        "summary_total_runtime.png",
    )
    save_bar_chart(
        peak_mem,
        "Peak Memory (MB)",
        "Peak Memory Comparison",
        "summary_peak_memory.png",
    )
