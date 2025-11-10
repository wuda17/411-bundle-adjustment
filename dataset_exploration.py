import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BALDataset:
    """
    Pure NumPy loader for Bundle Adjustment in the Large (BAL) dataset.
    """

    def __init__(self, folder_name):
        file_path = os.path.join("data", folder_name)
        self._load_bal(file_path)

    def _load_bal(self, file_path):
        with open(file_path, "r") as f:
            header = f.readline().strip().split()
            self.n_cameras = int(header[0])
            self.n_points = int(header[1])
            self.n_observations = int(header[2])

            # Observations: cam_idx, pt_idx, x, y
            obs = []
            for _ in range(self.n_observations):
                cam_idx, pt_idx, x, y = f.readline().strip().split()
                obs.append([int(cam_idx), int(pt_idx), float(x), float(y)])
            self.observations = np.array(obs)

            # Camera parameters: rotation(3), translation(3), intrinsics(3)
            cams = []
            for _ in range(self.n_cameras):
                cams.append(list(map(float, f.readline().strip().split())))
            self.camera_params = np.array(cams)

            # 3D points
            pts = []
            for _ in range(self.n_points):
                pts.append(list(map(float, f.readline().strip().split())))
            self.points_3d = np.array(pts)

    def get_properties(self):
        return {
            "n_cameras": self.n_cameras,
            "n_points": self.n_points,
            "n_observations": self.n_observations,
        }


# -------------------------------
# Reprojection and Optimization
# -------------------------------
def rodrigues_rotate_point(rvec, point):
    """Rotate 3D point using Rodrigues vector (axis-angle)."""
    theta = np.linalg.norm(rvec)
    if theta == 0:
        return point
    k = rvec / theta
    p = point
    return (
        p * np.cos(theta)
        + np.cross(k, p) * np.sin(theta)
        + k * np.dot(k, p) * (1 - np.cos(theta))
    )


def project_point(X, cam):
    """Project 3D point X using camera parameters."""
    rvec = cam[:3]
    tvec = cam[3:6]
    f, cx, cy = cam[6:9]
    X_rot = rodrigues_rotate_point(rvec, X)
    X_cam = X_rot + tvec
    x_proj = f * (X_cam[0] / X_cam[2]) + cx
    y_proj = f * (X_cam[1] / X_cam[2]) + cy
    return x_proj, y_proj


def reprojection_error(params, n_cameras, n_points, observations):
    cameras = params[: n_cameras * 9].reshape((n_cameras, 9))
    points = params[n_cameras * 9 :].reshape((n_points, 3))
    residuals = []
    for cam_idx, pt_idx, x_obs, y_obs in observations:
        X = points[pt_idx]
        cam = cameras[cam_idx]
        x_proj, y_proj = project_point(X, cam)
        residuals.extend([x_proj - x_obs, y_proj - y_obs])
    return np.array(residuals)


# -------------------------------
# Visualization
# -------------------------------
def visualize_scene(points_3d, camera_params, num_points=1000):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    pts = points_3d
    if pts.shape[0] > num_points:
        indices = np.random.choice(pts.shape[0], num_points, replace=False)
        pts = pts[indices]

    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="b", label="3D Points")

    cams = camera_params[:, 3:6]  # camera positions (translation)
    ax.scatter(
        cams[:, 0], cams[:, 1], cams[:, 2], s=20, c="r", marker="^", label="Cameras"
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    folder_name = "problem-49-7776-pre.txt"  # adjust path
    dataset = BALDataset(folder_name)
    print(dataset.get_properties())

    # Flatten all parameters for optimization
    x0 = np.hstack([dataset.camera_params.ravel(), dataset.points_3d.ravel()])

    # Visualize initial scene
    visualize_scene(dataset.points_3d, dataset.camera_params)

    # Example: Run optimization with scipy (optional)
    # from scipy.optimize import least_squares
    # result = least_squares(reprojection_error, x0, args=(dataset.n_cameras, dataset.n_points, dataset.observations))
    # optimized_cameras = result.x[:dataset.n_cameras*9].reshape((dataset.n_cameras,9))
    # optimized_points = result.x[dataset.n_cameras*9:].reshape((dataset.n_points,3))
    # visualize_scene(optimized_points, optimized_cameras)
