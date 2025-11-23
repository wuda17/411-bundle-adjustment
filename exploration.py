import time
import os
import shutil

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import numpy as np

import utils
import geometry
from metrics_logger import (
    SolverLogger,
    BenchmarkLogger,
    plot_iteration_metrics,
    plot_summary_comparison,
)


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9 :].reshape((n_points, 3))
    points_proj = geometry.project(
        points_3d[point_indices], camera_params[camera_indices]
    )
    return (points_proj - points_2d).ravel()


def fun_with_logging(
    params,
    n_cameras,
    n_points,
    cam_idx,
    pt_idx,
    points_2d,
    solver_logger,
    bench_logger,
    iter_counter,
):
    t0 = time.time()

    residuals = fun(params, n_cameras, n_points, cam_idx, pt_idx, points_2d)

    t1 = time.time()
    iteration_time = t1 - t0

    # Optional: Jacobian timing placeholder
    J = None
    jac_time = None

    # Log metrics
    solver_logger.log_iteration(iter_counter[0], params, residuals, J)
    bench_logger.log_iteration(iter_counter[0], iteration_time, jac_time)

    iter_counter[0] += 1
    return residuals


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """Compute the Jacobian sparsity pattern for bundle adjustment.

    Returns a sparse matrix indicating which parameters affect which residuals.
    This information helps the optimizer work more efficiently.
    """
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


if __name__ == "__main__":
    # ----------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------
    dataset = utils.BALDataset(
        "data/problem-257-65132-pre.txt.bz2", dataset_name="trafalgar"
    )

    n_cameras = dataset.n_cameras
    n_points = dataset.n_points
    cam_idx = dataset.camera_indices
    pt_idx = dataset.point_indices
    points_2d = dataset.points_2d
    camera_params = dataset.camera_params
    points_3d = dataset.points_3d

    print("\n=== Dataset Loaded ===")
    print(dataset.get_properties())

    # Initial parameter vector
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # ----------------------------------------------------------
    # Prepare output directory
    # ----------------------------------------------------------
    OUTPUT_DIR = "outputs"
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # Visualize BEFORE optimization
    # ----------------------------------------------------------
    print("\n=== Visualizing BEFORE optimization ===")
    utils.visualize_scene(points_3d, camera_params, frustum_scale_ratio=0.01)

    # ----------------------------------------------------------
    # Define experiments
    # ----------------------------------------------------------
    experiments = [
        # Note: runs into hard numeric limit on all BAL datasets
        # {
        #     "label": "Default LM",
        #     "jac_sparsity": None,
        #     "loss": "linear",
        #     "method": "trf",
        # },
        {
            "label": "Sparse LM",
            "jac_sparsity": bundle_adjustment_sparsity(
                n_cameras, n_points, cam_idx, pt_idx
            ),
            "loss": "linear",
            "method": "trf",
        },
        {
            "label": "Robust Cost",
            "jac_sparsity": bundle_adjustment_sparsity(
                n_cameras, n_points, cam_idx, pt_idx
            ),
            "loss": "soft_l1",
            "method": "trf",
        },
    ]

    solver_loggers = []
    bench_loggers = []

    # ---------------------------------------------------------
    # Run experiments
    # ---------------------------------------------------------
    for i, exp in enumerate(experiments, 1):

        print(f"\n=== Running Experiment {i}: {exp['label']} ===")

        solver_logger = SolverLogger(n_cameras, n_points, cam_idx, pt_idx)
        bench_logger = BenchmarkLogger()
        iter_counter = [0]

        # ---- RUN OPTIMIZATION -------------------------------------
        res = least_squares(
            fun_with_logging,
            x0,
            jac_sparsity=exp.get("jac_sparsity"),
            verbose=2,
            x_scale="jac",
            ftol=1e-4,
            method=exp["method"],
            loss=exp["loss"],
            args=(
                n_cameras,
                n_points,
                cam_idx,
                pt_idx,
                points_2d,
                solver_logger,
                bench_logger,
                iter_counter,
            ),
        )

        # ---------------------------------------------------------
        # Print summary metrics
        # ---------------------------------------------------------
        print("=== Solver / Problem Metrics ===")
        print(solver_logger.summary())
        print("=== Benchmarking Metrics ===")
        print(bench_logger.summary())

        # ---------------------------------------------------------
        # Extract optimized parameters
        # ---------------------------------------------------------
        camera_params_opt = res.x[: n_cameras * 9].reshape((n_cameras, 9))
        points_3d_opt = res.x[n_cameras * 9 :].reshape((n_points, 3))

        # ---------------------------------------------------------
        # Visualization
        # ---------------------------------------------------------
        utils.visualize_scene(
            points_3d,
            camera_params,
            title=f"After BA: {exp['label']}",
        )

        # ---------------------------------------------------------
        # Per-iteration diagnostic plots
        # ---------------------------------------------------------
        plot_iteration_metrics(
            solver_logger,
            bench_logger,
            experiment_number=i,
            label=exp["label"],
        )

        # Save for comparison charts later
        solver_loggers.append(solver_logger)
        bench_loggers.append(bench_logger)

    # --- Summary comparison across all experiments ---
    labels = [e["label"] for e in experiments]
    plot_summary_comparison(solver_loggers, bench_loggers, labels)
