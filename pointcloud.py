try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: Open3D not available. 3D point cloud visualization will be limited.")

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_point_cloud(depth_map, color_image, camera_params, max_depth=np.inf):
    """
    Create 3D point cloud from depth map and color image.
    """
    height, width = depth_map.shape
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']

    # Create meshgrid of pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten coordinates and depth
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    depth_flat = depth_map.flatten()

    # Filter valid depths
    valid_mask = np.isfinite(depth_flat) & (
        depth_flat > 0) & (depth_flat < max_depth)
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    depth_valid = depth_flat[valid_mask]

    # Convert to 3D coordinates
    X = (x_valid - cx) * depth_valid / fx
    Y = (y_valid - cy) * depth_valid / fy
    Z = depth_valid

    # Stack into points array
    points = np.column_stack((X, Y, Z))

    if OPEN3D_AVAILABLE:
        # Create Open3D point cloud
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # Add colors if color image is provided
        if color_image is not None:
            # Convert BGR to RGB and flatten
            color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            # Normalize to [0, 1]
            colors_flat = color_rgb.reshape(-1, 3) / 255.0

            # Filter colors for valid points
            colors_valid = colors_flat[valid_mask]
            point_cloud.colors = o3d.utility.Vector3dVector(colors_valid)

        return point_cloud, points
    else:
        # Return numpy array and colors for matplotlib visualization
        colors = None
        if color_image is not None:
            color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            colors_flat = color_rgb.reshape(-1, 3) / 255.0
            colors = colors_flat[valid_mask]

        return points, colors


def visualize_point_cloud(point_cloud_data, window_name="3D Point Cloud", save_path=None):
    """
    Visualize point cloud using available method.
    """
    if OPEN3D_AVAILABLE:
        point_cloud = point_cloud_data

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=800, height=600)

        vis.add_geometry(point_cloud)

        render_option = vis.get_render_option()
        render_option.point_size = 1.0
        render_option.background_color = np.asarray(
            [0.1, 0.1, 0.1])  # Dark background

        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])  # Look from front
        ctr.set_lookat([0, 0, 2])  # Look at center
        ctr.set_up([0, -1, 0])     # Up direction

        vis.run()
        vis.destroy_window()
    else:
        points, colors = point_cloud_data

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(window_name)

        max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                             points[:, 1].max()-points[:, 1].min(),
                             points[:, 2].max()-points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def save_point_cloud(point_cloud_data, filename):
    """
    Save point cloud to file.
    """
    if OPEN3D_AVAILABLE:
        point_cloud = point_cloud_data
        o3d.io.write_point_cloud(filename, point_cloud)
        print(f"Point cloud saved to {filename}")
    else:
        points, colors = point_cloud_data

        np.savez(filename.replace('.ply', '.npz'),
                 points=points, colors=colors)
        print(f"Point cloud saved to {filename.replace('.ply', '.npz')}")


def compute_point_cloud_statistics(point_cloud_data):
    """
    Compute statistics of the point cloud.
    """
    if OPEN3D_AVAILABLE:
        point_cloud = point_cloud_data
        points = np.asarray(point_cloud.points)
    else:
        points, _ = point_cloud_data

    print("\n=== Point Cloud Statistics ===")
    print(f"Number of points: {len(points)}")

    if len(points) > 0:
        print(
            f"X range: {np.min(points[:, 0]):.3f} to {np.max(points[:, 0]):.3f} m")
        print(
            f"Y range: {np.min(points[:, 1]):.3f} to {np.max(points[:, 1]):.3f} m")
        print(
            f"Z range: {np.min(points[:, 2]):.3f} to {np.max(points[:, 2]):.3f} m")

        center = points.mean(axis=0)
        print(
            f"Center of mass: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m")


def filter_point_cloud(point_cloud, min_bound=None, max_bound=None, nb_neighbors=20, std_ratio=2.0):
    """
    Filter point cloud to remove outliers.

    Parameters:
    - point_cloud: Input point cloud
    - min_bound: Minimum bounds [x, y, z]
    - max_bound: Maximum bounds [x, y, z]
    - nb_neighbors: Number of neighbors for statistical outlier removal
    - std_ratio: Standard deviation ratio for outlier removal

    Returns:
    - filtered_cloud: Filtered point cloud
    """
    filtered_cloud = point_cloud

    if min_bound is not None and max_bound is not None:
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound)
        filtered_cloud = filtered_cloud.crop(bbox)

    filtered_cloud, ind = filtered_cloud.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )

    print(
        f"Filtered point cloud: {len(filtered_cloud.points)} points (removed {len(point_cloud.points) - len(filtered_cloud.points)})")

    return filtered_cloud


def compare_point_clouds(pc1, pc2, label1="Point Cloud 1", label2="Point Cloud 2"):
    """
    Compare two point clouds.
    """
    points1 = np.asarray(pc1.points)
    points2 = np.asarray(pc2.points)

    print(f"\n=== Point Cloud Comparison ===")
    print(f"{label1}: {len(points1)} points")
    print(f"{label2}: {len(points2)} points")

    if len(points1) > 0 and len(points2) > 0:
        # Compute bounding boxes
        bbox1 = pc1.get_axis_aligned_bounding_box()
        bbox2 = pc2.get_axis_aligned_bounding_box()

        extent1 = bbox1.get_extent()
        extent2 = bbox2.get_extent()

        print(f"{label1} bounding box: {extent1}")
        print(f"{label2} bounding box: {extent2}")

        volume1 = extent1[0] * extent1[1] * extent1[2]
        volume2 = extent2[0] * extent2[1] * extent2[2]

        print(f"{label1} volume: {volume1:.3f} m³")
        print(f"{label2} volume: {volume2:.3f} m³")
        print(f"Volume ratio ({label2}/{label1}): {volume2/volume1:.3f}")
