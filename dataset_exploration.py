import bz2
import numpy as np
import matplotlib.pyplot as plt


class BALDataset:
    def __init__(self, file_path):
        self._load_bal(file_path)

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
        return {
            "n_cameras": self.n_cameras,
            "n_points": self.n_points,
            "n_observations": self.n_observations,
        }


# -------------------------------
# Visualization
# -------------------------------
import matplotlib.pyplot as plt
import numpy as np


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
