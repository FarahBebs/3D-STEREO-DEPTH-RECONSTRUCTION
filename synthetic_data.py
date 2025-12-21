import numpy as np
import cv2
import matplotlib.pyplot as plt


def generate_synthetic_stereo_pair(image_path=None, shift_pixels=50, baseline=0.1, focal_length=500):
    """
    Generate synthetic stereo pair by shifting an image horizontally.

    Parameters:
    - image_path: Path to base image. If None, creates a simple synthetic scene.
    - shift_pixels: Number of pixels to shift for disparity.
    - baseline: Baseline distance in meters.
    - focal_length: Focal length in pixels.

    Returns:
    - left_image: Left stereo image
    - right_image: Right stereo image
    - ground_truth_disparity: Known disparity map
    - camera_params: Dictionary with camera intrinsics and extrinsics
    """
    if image_path is None:
        height, width = 400, 600
        left_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Create depth layers
        depths = [2.0, 3.0, 4.0, 5.0]  # depths in meters
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR

        for i, (depth, color) in enumerate(zip(depths, colors)):
            start_y = i * 100
            end_y = (i + 1) * 100
            start_x = 50 + i * 50
            end_x = start_x + 100
            left_image[start_y:end_y, start_x:end_x] = color

        for y in range(0, height, 20):
            cv2.line(left_image, (0, y), (width, y), (128, 128, 128), 1)
        for x in range(0, width, 20):
            cv2.line(left_image, (x, 0), (x, height), (128, 128, 128), 1)
    else:
        left_image = cv2.imread(image_path)
        if left_image is None:
            raise ValueError(f"Could not load image from {image_path}")

    height, width = left_image.shape[:2]

    right_image = np.zeros_like(left_image)
    right_image[:, shift_pixels:] = left_image[:, :-shift_pixels]

    right_image[:, :shift_pixels] = left_image[:, :shift_pixels]

    ground_truth_disparity = np.zeros((height, width), dtype=np.float32)
    ground_truth_disparity[:, shift_pixels:] = shift_pixels

    camera_params = {
        'focal_length': focal_length,  # fx = fy = focal_length
        'cx': width // 2,
        'cy': height // 2,
        'baseline': baseline,
        'fx': focal_length,
        'fy': focal_length
    }

    return left_image, right_image, ground_truth_disparity, camera_params


def visualize_synthetic_data(left_image, right_image, ground_truth_disparity, save_path=None):
    """
    Visualize the synthetic stereo pair and ground truth disparity.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Left Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Right Image')
    axes[1].axis('off')

    disp_vis = axes[2].imshow(ground_truth_disparity, cmap='gray')
    axes[2].set_title('Ground Truth Disparity')
    plt.colorbar(disp_vis, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    left, right, gt_disp, cam_params = generate_synthetic_stereo_pair()

    print("Camera Parameters:")
    for key, value in cam_params.items():
        print(f"  {key}: {value}")

    # Visualize
    visualize_synthetic_data(left, right, gt_disp)
