import cv2
import numpy as np
from synthetic_data import generate_synthetic_stereo_pair


def load_stereo_images(left_path=None, right_path=None, use_synthetic=True):
    """
    Load stereo image pair. If paths are None, generate synthetic data.

    Parameters:
    - left_path: Path to left image
    - right_path: Path to right image
    - use_synthetic: If True and paths are None, generate synthetic data

    Returns:
    - left_image: Left stereo image (BGR)
    - right_image: Right stereo image (BGR)
    - camera_params: Dictionary with camera intrinsics/extrinsics
    """
    if left_path is not None and right_path is not None:
        left_image = cv2.imread(left_path)
        right_image = cv2.imread(right_path)

        if left_image is None or right_image is None:
            raise ValueError("Could not load one or both images")

        # For real images, use synthetic camera parameters
        # In a real application, these would come from calibration
        height, width = left_image.shape[:2]
        camera_params = {
            'focal_length': 500,  # pixels
            'cx': width // 2,
            'cy': height // 2,
            'baseline': 0.1,  # meters
            'fx': 500,
            'fy': 500
        }

        return left_image, right_image, camera_params

    elif use_synthetic:
        left_image, right_image, _, camera_params = generate_synthetic_stereo_pair()
        return left_image, right_image, camera_params

    else:
        raise ValueError("No image paths provided and synthetic data disabled")


def rectify_stereo_pair(left_image, right_image, camera_params):
    """
    Rectify stereo image pair. For synthetic data, assume already rectified.
    In a real scenario, this would use calibration matrices.

    Parameters:
    - left_image: Left stereo image
    - right_image: Right stereo image
    - camera_params: Camera parameters dictionary

    Returns:
    - left_rectified: Rectified left image
    - right_rectified: Rectified right image
    - Q: Disparity-to-depth mapping matrix (for OpenCV compatibility)
    """

    left_rectified = left_image.copy()
    right_rectified = right_image.copy()

    # Create Q matrix for disparity to depth conversion
    # Q is the disparity-to-depth mapping matrix
    # For rectified stereo, Q has the form:
    # [[1, 0, 0, -cx],
    #  [0, 1, 0, -cy],
    #  [0, 0, 0, fx],
    #  [0, 0, -1/baseline, 0]]

    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['cx']
    cy = camera_params['cy']
    baseline = camera_params['baseline']

    Q = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1/baseline, 0]
    ], dtype=np.float32)

    return left_rectified, right_rectified, Q


def visualize_rectified_images(left_rectified, right_rectified, save_path=None):
    """
    Display rectified stereo images side by side.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Rectified Left Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(right_rectified, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Rectified Right Image')
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
