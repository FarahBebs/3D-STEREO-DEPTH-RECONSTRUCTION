import numpy as np
import cv2
from scipy.optimize import minimize_scalar
import time


def compute_disparity_block_matching(left_image, right_image, block_size=15, max_disparity=64):
    """
    Compute disparity map using block matching with Sum of Squared Differences (SSD).

    Parameters:
    - left_image: Rectified left image (grayscale)
    - right_image: Rectified right image (grayscale)
    - block_size: Size of the matching block (odd number)
    - max_disparity: Maximum disparity to search

    Returns:
    - disparity_map: Disparity map (pixels)
    """
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_image
        right_gray = right_image

    height, width = left_gray.shape

    disparity_map = np.zeros((height, width), dtype=np.float32)

    half_block = block_size // 2

    for y in range(half_block, height - half_block):
        for x in range(half_block, width - half_block):

            left_block = left_gray[y - half_block:y + half_block + 1,
                                   x - half_block:x + half_block + 1]

            min_ssd = float('inf')
            best_disparity = 0

            for d in range(max_disparity + 1):
                if x - d - half_block < 0:
                    continue

                right_block = right_gray[y - half_block:y + half_block + 1,
                                         x - d - half_block:x - d + half_block + 1]

                ssd = np.sum((left_block - right_block) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    return disparity_map


def ssd_cost_function(disparity, left_block, right_image, x, y, half_block):
    """
    SSD cost function for optimization.

    Parameters:
    - disparity: Current disparity value
    - left_block: Left image block
    - right_image: Right image
    - x, y: Block center coordinates
    - half_block: Half block size

    Returns:
    - ssd: Sum of squared differences
    """
    d = int(disparity)
    if x - d - half_block < 0 or x - d + half_block + 1 > right_image.shape[1]:
        return float('inf')  # Penalize out of bounds

    right_block = right_image[y - half_block:y + half_block + 1,
                              x - d - half_block:x - d + half_block + 1]

    ssd = np.sum((left_block - right_block) ** 2)
    return ssd


def compute_disparity_optimization(left_image, right_image, block_size=15, max_disparity=64,
                                   method='gradient_descent', num_pixels=1000):
    """
    Compute disparity map using numerical optimization.

    Parameters:
    - left_image: Rectified left image (grayscale)
    - right_image: Rectified right image (grayscale)
    - block_size: Size of the matching block
    - max_disparity: Maximum disparity to search
    - method: Optimization method ('gradient_descent', 'golden_section')
    - num_pixels: Number of pixels to optimize (for performance)

    Returns:
    - disparity_map: Disparity map
    - error_history: List of error values during optimization
    - iteration_count: Number of iterations
    """
    if len(left_image.shape) == 3:
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_image
        right_gray = right_image

    height, width = left_gray.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)

    half_block = block_size // 2

    np.random.seed(42)
    pixels_to_optimize = []

    for _ in range(num_pixels):
        y = np.random.randint(half_block, height - half_block)
        x = np.random.randint(half_block + max_disparity, width - half_block)
        pixels_to_optimize.append((y, x))

    total_error_history = []
    total_iterations = 0

    for y, x in pixels_to_optimize:
        # Extract left block
        left_block = left_gray[y - half_block:y + half_block + 1,
                               x - half_block:x + half_block + 1]

        # Define the cost function for this pixel
        def cost(d):
            return ssd_cost_function(d, left_block, right_gray, x, y, half_block)

        if method == 'gradient_descent':

            d_current = max_disparity // 2
            learning_rate = 0.1
            max_iter = 100
            tolerance = 1e-3

            error_history = []
            for iteration in range(max_iter):
                error = cost(d_current)
                error_history.append(error)

                grad = (cost(d_current + 1) - cost(d_current - 1)) / 2

                if abs(grad) < tolerance:
                    break

                d_current = np.clip(
                    d_current - learning_rate * grad, 0, max_disparity)

            optimal_d = d_current
            iterations = len(error_history)

        elif method == 'golden_section':

            result = minimize_scalar(cost, bounds=(
                0, max_disparity), method='golden')
            optimal_d = result.x
            iterations = result.nfev
            error_history = [cost(optimal_d)]  # Simplified

        else:
            raise ValueError(f"Unknown optimization method: {method}")

        disparity_map[y, x] = optimal_d
        total_error_history.extend(error_history)
        total_iterations += iterations

    return disparity_map, total_error_history, total_iterations


def visualize_disparity_map(disparity_map, title="Disparity Map", save_path=None):
    """
    Visualize disparity map as grayscale image.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 8))
    plt.imshow(disparity_map, cmap='gray')
    plt.colorbar(label='Disparity (pixels)')
    plt.title(title)
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def save_disparity_map(disparity_map, filename):
    """
    Save disparity map as image file.
    """
    disp_norm = cv2.normalize(disparity_map, None, 0,
                              255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(filename, disp_norm)
