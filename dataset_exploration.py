import os
import bz2
import urllib.request
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


class BALDataset:
    _DATASET_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"

    def __init__(self, file_path: str):
        # If no file path is provided, download default
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.download_dataset(file_name=self.file_path.name)
        self._load_bal(self.file_path)

    @staticmethod
    def download_dataset(file_name="problem-49-7776-pre.txt.bz2"):
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir / file_name
        if not file_path.exists():
            url = BALDataset._DATASET_URL + file_name
            print(f"Downloading {file_name} from {url}...")
            urllib.request.urlretrieve(url, file_path)
            print("Download complete.")
        else:
            print(f"File {file_name} already exists, skipping download.")
        return

    def _load_bal(self, file_path):
        with bz2.open(file_path, "rt") as file:
            # Header
            n_cameras, n_points, n_observations = map(int, file.readline().split())
            self.n_cameras = n_cameras
            self.n_points = n_points
            self.n_observations = n_observations

            # Observations
            camera_indices = np.empty(n_observations, dtype=int)
            point_indices = np.empty(n_observations, dtype=int)
            points_2d = np.empty((n_observations, 2), dtype=np.float64)
            for i in range(n_observations):
                cam_idx, pt_idx, x, y = file.readline().split()
                camera_indices[i] = int(cam_idx)
                point_indices[i] = int(pt_idx)
                points_2d[i] = [float(x), float(y)]
            self.camera_indices = camera_indices
            self.point_indices = point_indices
            self.points_2d = points_2d

            # Camera parameters
            camera_params = np.empty(n_cameras * 9, dtype=np.float64)
            for i in range(n_cameras * 9):
                camera_params[i] = float(file.readline())
            self.camera_params = camera_params.reshape((n_cameras, 9))

            # 3D points
            points_3d = np.empty(n_points * 3, dtype=np.float64)
            for i in range(n_points * 3):
                points_3d[i] = float(file.readline())
            self.points_3d = points_3d.reshape((n_points, 3))

    def get_properties(self):
        """
        Returns dataset properties including:
        - Number of cameras
        - Number of 3D points
        - Number of observations
        - Total number of parameters (for optimization)
        - Total number of residuals
        """
        n_cameras = self.camera_params.shape[0]
        n_points = self.points_3d.shape[0]

        # Total parameters for bundle adjustment:
        # 9 parameters per camera (rotation + translation + intrinsics)
        # 3 parameters per 3D point
        n = 9 * n_cameras + 3 * n_points

        # Total residuals = 2 * number of 2D observations
        m = 2 * self.points_2d.shape[0]

        return {
            "n_cameras": n_cameras,
            "n_points": n_points,
            "n_observations": self.n_observations,
            "total_parameters": n,
            "total_residuals": m,
        }


# TODO: fix
def visualize_scene(points_3d, camera_params, num_points=10_000):
    """
    Visualize 3D points and camera positions from a BAL dataset.

    Coordinate system notes:
    - points_3d: right-handed world coordinates (X right, Y up, Z forward)
    - camera_params: rotation (rvec) + translation (tvec) + intrinsics
    - For visualization, we flip the Y-axis to match typical 3D plotting convention.
    - Camera positions are shown at tvec (world frame)
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample points if dataset is large
    pts = points_3d.copy()
    if pts.shape[0] > num_points:
        indices = np.random.choice(pts.shape[0], num_points, replace=False)
        pts = pts[indices]

    # Flip Y-axis for standard 3D visualization
    pts[:, 1] *= -1

    # Plot 3D points
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2], s=1, c="b", label="3D Points (world frame)"
    )

    # Camera positions: translation vector in world coordinates
    cams = camera_params[:, 3:6].copy()
    cams[:, 1] *= -1  # flip Y-axis same as points
    ax.scatter(
        cams[:, 0],
        cams[:, 1],
        cams[:, 2],
        s=20,
        c="r",
        marker="^",
        label="Cameras (world frame)",
    )

    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z (forward)")
    ax.set_title("3D Scene Reconstruction (BAL Dataset)")
    ax.legend()
    plt.show()


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    dataset = BALDataset("data/problem-49-7776-pre.txt.bz2")
    print(dataset.get_properties())
    print("Points shape:", dataset.points_3d.shape)
    print("Camera params shape:", dataset.camera_params.shape)
    visualize_scene(dataset.points_3d, dataset.camera_params)
