import bz2
import urllib.request
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


class BALDataset:
    _DATASET_URL = "http://grail.cs.washington.edu/projects/bal/data/"

    def __init__(self, file_path: str, dataset_name: str = None):
        # If no file path is provided, download default
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.download_dataset(
                file_name=self.file_path.name, dataset_name=dataset_name
            )
        self._load_bal(self.file_path)

    @staticmethod
    def download_dataset(file_name, dataset_name=None):
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        file_path = data_dir / file_name
        if not file_path.exists():
            url = f"{BALDataset._DATASET_URL}/{dataset_name}/{file_name}"
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


def visualize_scene(
    points_3d,
    camera_params,
    num_points=10000,
    frustum_size=0.1,  # static constant frustum size
    title="3D Scene",
    clip_percentile=(1, 99),
    max_frustum_distance=10.0,  # absolute cutoff (after centering)
):
    """
    Visualize BAL point cloud + camera frustums with a static frustum size.
    The point cloud is centered but not scaled, so frustums remain constant.
    """

    pts = points_3d.copy()

    # ---- Clip extreme Z outliers ----
    z = pts[:, 2]
    z_min, z_max = np.percentile(z, clip_percentile)
    pts = pts[(z >= z_min) & (z <= z_max)]

    # ---- Subsample ----
    if pts.shape[0] > num_points:
        rng = np.random.default_rng(42)
        pts = pts[rng.choice(pts.shape[0], num_points, replace=False)]

    # Flip Y for visualization convention
    pts[:, 1] *= -1

    # ---- Compute camera centers ----
    cams = []
    rotations = []
    for cam in camera_params:
        rvec, tvec = cam[:3], cam[3:6]
        theta = np.linalg.norm(rvec)

        if theta < 1e-8:
            R = np.eye(3)
        else:
            v = rvec / theta
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

        C = -R.T @ tvec
        C[1] *= -1
        cams.append(C)
        rotations.append(R)

    cams = np.vstack(cams)
    rotations = np.array(rotations)

    # ---- Center scene around point cloud (not scaled) ----
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    cams_centered = cams - centroid

    # --- Scale cloud ---
    # --- Depth gradient coloring ---
    zc = pts_centered[:, 2]
    colors = plt.cm.viridis((zc - zc.min()) / (zc.max() - zc.min()))[:, :3]

    # Create point cloud geometry
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_centered)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # ---- Camera frustums with STATIC size ----
    frustums = []
    for i, R in enumerate(rotations):
        C = cams_centered[i]

        # Skip outlier cameras
        if np.linalg.norm(C) > max_frustum_distance:
            continue

        # Frustum corners (constant size)
        w = h = 0.5 * frustum_size
        zf = frustum_size

        corners_cam = np.array(
            [
                [w, h, zf],
                [w, -h, zf],
                [-w, -h, zf],
                [-w, h, zf],
            ]
        )

        # Transform to world
        corners_world = (R.T @ corners_cam.T).T + C

        # Lines from center to corners + between corners
        points = np.vstack([C, corners_world])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(points)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))

        frustums.append(ls)

    # ---- Visualization ----
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    for f in frustums:
        vis.add_geometry(f)

    # View settings
    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()
    ctr.set_lookat([0, 0, 0])
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, 1, 0])
    ctr.set_zoom(0.7)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Test dataset loading and visualization
    dataset = BALDataset("data/problem-951-708276-pre.txt.bz2")
    print(dataset.get_properties())
    visualize_scene(dataset.points_3d, dataset.camera_params)
