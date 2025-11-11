import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import utils
import geometry


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
    # Load the BAL dataset
    dataset = utils.BALDataset("data/problem-52-64053-pre.txt.bz2")
    props = dataset.get_properties()
    print("Dataset properties:")
    print(props)
    print()

    # Extract dataset information
    n_cameras = dataset.n_cameras
    n_points = dataset.n_points
    camera_indices = dataset.camera_indices
    point_indices = dataset.point_indices
    points_2d = dataset.points_2d
    camera_params = dataset.camera_params
    points_3d = dataset.points_3d

    # Visualize the 3D scene
    print("Visualizing 3D scene...")
    utils.visualize_scene(points_3d, camera_params)

    # Prepare initial parameters for optimization
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    # Compute and visualize initial residuals
    print("\nComputing initial residuals...")
    f0 = geometry.fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(f0)
    plt.title("Initial Residuals (Before Optimization)")
    plt.xlabel("Residual Index")
    plt.ylabel("Residual Value")
    plt.grid(True, alpha=0.3)
    print(f"Initial cost (sum of squared residuals): {np.sum(f0**2):.4e}")

    # Compute the Jacobian sparsity pattern
    print("\nComputing Jacobian sparsity pattern...")
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)

    # Run bundle adjustment optimization
    print("\nRunning bundle adjustment optimization...")
    print("=" * 70)
    t0 = time.time()
    res = least_squares(
        geometry.fun,
        x0,
        jac_sparsity=A,
        verbose=2,
        x_scale="jac",
        ftol=1e-4,
        method="trf",
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d),
    )
    t1 = time.time()
    print("=" * 70)
    print(f"\nOptimization took {t1 - t0:.0f} seconds")
    print(f"Initial cost: {np.sum(f0**2):.4e}")
    print(f"Final cost: {np.sum(res.fun**2):.4e}")
    print(f"Number of function evaluations: {res.nfev}")
    print()

    # Visualize optimized residuals
    plt.subplot(1, 2, 2)
    plt.plot(res.fun)
    plt.title("Final Residuals (After Optimization)")
    plt.xlabel("Residual Index")
    plt.ylabel("Residual Value")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Extract optimized parameters
    camera_params_opt = res.x[: n_cameras * 9].reshape((n_cameras, 9))
    points_3d_opt = res.x[n_cameras * 9 :].reshape((n_points, 3))

    # Visualize the optimized 3D scene
    print("Visualizing optimized 3D scene...")
    utils.visualize_scene(points_3d_opt, camera_params_opt)
