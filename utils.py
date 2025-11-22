import bz2
import urllib.request
from pathlib import Path

import numpy as np
import open3d as o3d


class BALDataset:
    _DATASET_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"

    def __init__(self, file_path: str):
        # If no file path is provided, download default
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            self.download_dataset(file_name=self.file_path.name)
        self._load_bal(self.file_path)

    @staticmethod
    def download_dataset(file_name):
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


def visualize_scene(points_3d, camera_params, num_points=10000, frustum_scale=0.5):
    """
    Visualize BAL 3D points and cameras as frustums in Open3D.

    points_3d: Nx3 array
    camera_params: Nx9 array [rx, ry, rz, tx, ty, tz, f, k1, k2]
    """
    # Subsample points if too large
    pts = points_3d.copy()
    if pts.shape[0] > num_points:
        idx = np.random.choice(pts.shape[0], num_points, replace=False)
        pts = pts[idx]

    # Flip Y-axis for standard 3D visualization
    pts[:, 1] *= -1

    # Compute camera centers in world coordinates
    cams = []
    for cam in camera_params:
        rvec = cam[:3]
        tvec = cam[3:6]
        theta = np.linalg.norm(rvec)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            v = rvec / theta
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        C = -R.T @ tvec
        C[1] *= -1  # flip Y
        cams.append(C)
    cams = np.vstack(cams)

    # Center scene around combined centroid
    centroid = np.mean(np.vstack([pts, cams]), axis=0)
    pts_centered = pts - centroid
    cams_centered = cams - centroid

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_centered)
    pcd.paint_uniform_color([0, 0.5, 1.0])  # blue

    # Camera frustums as small pyramids
    frustums = []
    for i, cam in enumerate(camera_params):
        C = cams_centered[i]
        rvec = cam[:3]
        tvec = cam[3:6]
        theta = np.linalg.norm(rvec)
        if theta < 1e-8:
            R = np.eye(3)
        else:
            v = rvec / theta
            K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

        # Frustum corners in camera frame (looking down +Z axis)
        f = cam[6]
        width = 0.1  # arbitrary frustum width
        height = 0.1  # arbitrary frustum height
        z = frustum_scale
        corners = np.array(
            [
                [width, height, z],
                [width, -height, z],
                [-width, -height, z],
                [-width, height, z],
            ]
        )
        # Transform corners to world using R.T (camera-to-world rotation)
        corners_world = (R.T @ corners.T).T + C
        # Create lines from camera center to corners and base edges
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
        # add center point
        points = np.vstack([C, corners_world])
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
        frustums.append(line_set)

    # Open3D visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    for f in frustums:
        vis.add_geometry(f)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0))
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # Test dataset loading and visualization
    dataset = BALDataset("data/problem-951-708276-pre.txt.bz2")
    print(dataset.get_properties())
    visualize_scene(dataset.points_3d, dataset.camera_params)
