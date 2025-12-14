import numpy as np
import cv2
import matplotlib.pyplot as plt


def disparity_to_depth(disparity_map, focal_length, baseline, max_depth=50.0):
    """
    Convert disparity map to depth map.

    Mathematical formula:
    depth = (focal_length * baseline) / disparity

    Where:
    - focal_length: camera focal length in pixels
    - baseline: distance between cameras in meters
    - disparity: disparity in pixels

    Parameters:
    - disparity_map: Disparity map (pixels)
    - focal_length: Focal length in pixels
    - baseline: Baseline in meters
    - max_depth: Maximum valid depth in meters

    Returns:
    - depth_map: Depth map in meters
    """
    # Avoid division by zero and set minimum disparity
    disparity_map = np.where(disparity_map < 1e-6, 1e-6, disparity_map)

    # Convert disparity to depth
    depth_map = (focal_length * baseline) / disparity_map

    # Set maximum depth limit
    depth_map = np.where(depth_map > max_depth, max_depth, depth_map)

    return depth_map


def visualize_depth_map(depth_map, title="Depth Map", max_depth=None, save_path=None):
    """
    Visualize depth map as a heatmap.
    """
    # Replace infinite depths with a large value for visualization
    depth_vis = np.where(np.isinf(depth_map), np.nanmax(
        depth_map[np.isfinite(depth_map)]), depth_map)

    if max_depth is None:
        # Use 95th percentile to avoid outliers
        max_depth = np.nanpercentile(depth_vis, 95)

    plt.figure(figsize=(10, 8))

    # Create heatmap
    im = plt.imshow(depth_vis, cmap='plasma_r', vmin=0, vmax=max_depth)

    # Add colorbar
    cbar = plt.colorbar(im, label='Depth (meters)')
    cbar.set_ticks(np.linspace(0, max_depth, 6))

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_depth_histogram(depth_map, bins=50, title="Depth Distribution", save_path=None):
    """
    Create histogram of depth values.
    """
    # Filter out infinite depths
    finite_depths = depth_map[np.isfinite(depth_map)]

    plt.figure(figsize=(10, 6))
    plt.hist(finite_depths, bins=bins, alpha=0.7,
             color='blue', edgecolor='black')
    plt.xlabel('Depth (meters)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def compute_depth_accuracy(predicted_depth, ground_truth_depth, mask=None):
    """
    Compute depth accuracy metrics.

    Parameters:
    - predicted_depth: Predicted depth map
    - ground_truth_depth: Ground truth depth map
    - mask: Optional mask for valid pixels

    Returns:
    - metrics: Dictionary with accuracy metrics
    """
    if mask is None:
        mask = np.isfinite(predicted_depth) & np.isfinite(ground_truth_depth)
    else:
        mask = mask & np.isfinite(
            predicted_depth) & np.isfinite(ground_truth_depth)

    pred = predicted_depth[mask]
    gt = ground_truth_depth[mask]

    if len(pred) == 0:
        return {'error': np.nan, 'rmse': np.nan, 'mae': np.nan, 'rel_error': np.nan}

    # Absolute error
    abs_error = np.abs(pred - gt)

    # Root Mean Square Error
    rmse = np.sqrt(np.mean((pred - gt) ** 2))

    # Mean Absolute Error
    mae = np.mean(abs_error)

    # Relative error (percentage)
    rel_error = np.mean(abs_error / gt) * 100

    # Threshold accuracy (delta < 1.25)
    delta = np.maximum(pred / gt, gt / pred)
    acc_125 = np.mean(delta < 1.25) * 100

    metrics = {
        'rmse': rmse,
        'mae': mae,
        'rel_error': rel_error,
        'acc_125': acc_125,
        'mean_error': np.mean(pred - gt),
        'median_error': np.median(pred - gt)
    }

    return metrics


def print_depth_statistics(depth_map, title="Depth Statistics"):
    """
    Print statistical summary of depth map.
    """
    finite_depths = depth_map[np.isfinite(depth_map)]

    print(f"\n=== {title} ===")
    print(
        f"Valid pixels: {len(finite_depths)} / {depth_map.size} ({100*len(finite_depths)/depth_map.size:.1f}%)")
    print(f"Min depth: {np.min(finite_depths):.3f} m")
    print(f"Max depth: {np.max(finite_depths):.3f} m")
    print(f"Mean depth: {np.mean(finite_depths):.3f} m")
    print(f"Median depth: {np.median(finite_depths):.3f} m")
    print(
        f"Depth range: {np.max(finite_depths) - np.min(finite_depths):.3f} m")


def compare_depth_maps(depth1, depth2, label1="Method 1", label2="Method 2", ground_truth=None):
    """
    Compare two depth maps and optionally against ground truth.
    """
    print(f"\n=== Depth Map Comparison: {label1} vs {label2} ===")

    # Statistics for both maps
    print_depth_statistics(depth1, f"{label1} Statistics")
    print_depth_statistics(depth2, f"{label2} Statistics")

    if ground_truth is not None:
        print_depth_statistics(ground_truth, "Ground Truth Statistics")

        # Accuracy metrics
        metrics1 = compute_depth_accuracy(depth1, ground_truth)
        metrics2 = compute_depth_accuracy(depth2, ground_truth)

        print(f"\n{label1} vs Ground Truth:")
        print(f"  RMSE: {metrics1['rmse']:.4f} m")
        print(f"  MAE: {metrics1['mae']:.4f} m")
        print(f"  Rel Error: {metrics1['rel_error']:.2f}%")
        print(f"  δ < 1.25: {metrics1['acc_125']:.1f}%")

        print(f"\n{label2} vs Ground Truth:")
        print(f"  RMSE: {metrics2['rmse']:.4f} m")
        print(f"  MAE: {metrics2['mae']:.4f} m")
        print(f"  Rel Error: {metrics2['rel_error']:.2f}%")
        print(f"  δ < 1.25: {metrics2['acc_125']:.1f}%")

        return metrics1, metrics2

    return None, None
