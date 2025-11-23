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
    frustum_scale_ratio=0.05,  # fraction of cloud bbox size
    title="3D Scene",
    clip_percentile=(1, 99),
    max_frustum_distance_ratio=3.0,  # skip frustums beyond this factor of cloud bbox
):
    """
    Visualize BAL 3D points and cameras with robust zoom and skipped far-away frustums.

    Args:
        points_3d: Nx3 array of 3D points
        camera_params: Mx9 array [rx, ry, rz, tx, ty, tz, f, k1, k2]
        num_points: max points to visualize
        frustum_scale_ratio: scale of frustums relative to cloud size
        title: window title
        clip_percentile: clip extreme points (min%, max%) in Z
        max_frustum_distance_ratio: skip frustums further than this times cloud bbox
    """
    pts = points_3d.copy()

    # --- Clip outliers ---
    z = pts[:, 2]
    z_min, z_max = np.percentile(z, clip_percentile)
    pts = pts[(z >= z_min) & (z <= z_max)]

    # --- Subsample ---
    if pts.shape[0] > num_points:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(pts.shape[0], num_points, replace=False)
        pts = pts[idx]

    pts[:, 1] *= -1  # Flip Y

    # --- Compute camera centers ---
    cams = []
    for cam in camera_params:
        rvec, tvec = cam[:3], cam[3:6]
        theta = np.linalg.norm(rvec)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            v = rvec / theta
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        C = -R.T @ tvec
        C[1] *= -1
        cams.append(C)
    cams = np.vstack(cams)

    # --- Center cloud only ---
    centroid = np.mean(pts, axis=0)
    pts_centered = pts - centroid
    cams_centered = cams - centroid

    # --- Scale cloud ---
    bbox_size = np.max(pts_centered.max(axis=0) - pts_centered.min(axis=0))
    pts_centered /= bbox_size
    cams_centered /= bbox_size

    # --- Depth gradient coloring ---
    zc = pts_centered[:, 2]
    colors = plt.cm.viridis((zc - zc.min()) / (zc.max() - zc.min()))[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_centered)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # --- Camera frustums (skip distant ones) ---
    frustums = []
    frustum_size = frustum_scale_ratio
    max_dist = max_frustum_distance_ratio  # factor of bbox
    for i, cam in enumerate(camera_params):
        C = cams_centered[i]
        if np.linalg.norm(C) > max_dist:  # skip far frustums
            continue

        rvec, tvec = cam[:3], cam[3:6]
        theta = np.linalg.norm(rvec)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            v = rvec / theta
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # Frustum corners
        width = height = 0.1 * frustum_size
        zf = frustum_size
        corners = np.array(
            [
                [width, height, zf],
                [width, -height, zf],
                [-width, -height, zf],
                [-width, height, zf],
            ]
        )
        corners_world = (R.T @ corners.T).T + C

        points = np.vstack([C, corners_world])
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
        frustums.append(line_set)

    # --- Visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    for f in frustums:
        vis.add_geometry(f)

    # Optional: small coordinate frame
    # vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    # --- Set view ---
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
