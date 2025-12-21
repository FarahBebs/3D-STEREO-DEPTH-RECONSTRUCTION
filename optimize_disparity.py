import numpy as np
import cv2
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import time


def ssd_residuals(disparity, left_block, right_image, x, y, half_block):
    """
    Compute SSD residuals for Levenberg-Marquardt optimization.

    Parameters:
    - disparity: Array with single disparity value
    - left_block: Left image block (flattened)
    - right_image: Right image
    - x, y: Block center coordinates
    - half_block: Half block size

    Returns:
    - residuals: Difference between left and right blocks
    """
    d = int(disparity[0])
    if x - d - half_block < 0 or x - d + half_block + 1 > right_image.shape[1]:

        return np.full_like(left_block, 1000)

    right_block = right_image[y - half_block:y + half_block + 1,
                              x - d - half_block:x - d + half_block + 1]

    residuals = left_block - right_block.flatten()
    return residuals


def optimize_disparity_levenberg_marquardt(left_image, right_image, block_size=15,
                                           max_disparity=64, num_pixels=500):
    """
    Compute disparity using Levenberg-Marquardt optimization.

    Parameters:
    - left_image: Rectified left image (grayscale)
    - right_image: Rectified right image (grayscale)
    - block_size: Size of matching block
    - max_disparity: Maximum disparity
    - num_pixels: Number of pixels to optimize

    Returns:
    - disparity_map: Optimized disparity map
    - error_history: Error vs iteration data
    - convergence_info: Optimization results
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

    all_error_histories = []
    convergence_messages = []
    successful_optimizations = 0

    for y, x in pixels_to_optimize:

        left_block = left_gray[y - half_block:y + half_block + 1,
                               x - half_block:x + half_block + 1].flatten()

        x0 = np.array([max_disparity // 2])

        def residuals(d):
            return ssd_residuals(d, left_block, right_gray, x, y, half_block)

        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=50)

            optimal_d = np.clip(result.x[0], 0, max_disparity)
            disparity_map[y, x] = optimal_d

            final_error = np.sum(result.fun ** 2)
            all_error_histories.append(final_error)

            if result.success:
                successful_optimizations += 1
                convergence_messages.append(f"Pixel ({x},{y}): Converged")
            else:
                convergence_messages.append(
                    f"Pixel ({x},{y}): {result.message}")

        except Exception as e:
            # Fallback to initial guess
            disparity_map[y, x] = x0[0]
            all_error_histories.append(float('inf'))
            convergence_messages.append(
                f"Pixel ({x},{y}): Optimization failed - {str(e)}")

    # Create error history (simplified - just final errors)
    error_history = all_error_histories if all_error_histories else [0]

    return disparity_map, all_error_histories, convergence_messages, successful_optimizations


def plot_optimization_error(error_history, title="Optimization Error vs Iteration", save_path=None):
    """
    Plot error vs iteration for the optimization process.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(error_history, 'b-', linewidth=2, markersize=4, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('SSD Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def print_optimization_summary(error_history, convergence_messages, runtime, successful_optimizations):
    """
    Print optimization summary statistics.
    """
    print("\n=== Optimization Summary ===")
    print(f"Runtime: {runtime:.3f} seconds")
    print(f"Number of optimized pixels: {len(error_history)}")
    print(
        f"Final errors range: {min(error_history):.2f} - {max(error_history):.2f}")
    print(f"Mean final error: {np.mean(error_history):.2f}")
    print(f"Median final error: {np.median(error_history):.2f}")

    success_count = successful_optimizations
    print(
        f"Convergence rate: {success_count}/{len(convergence_messages)} ({100*success_count/len(convergence_messages):.1f}%)")

    print("\nSample convergence messages:")
    for i, msg in enumerate(convergence_messages[:5]):
        print(f"  {i+1}. {msg}")


def compare_optimization_methods(left_image, right_image, block_size=15, max_disparity=64):
    """
    Compare different optimization methods.
    """
    methods = ['gradient_descent', 'golden_section']
    results = {}

    for method in methods:
        print(f"\nRunning {method} optimization...")
        start_time = time.time()

        from compute_disparity import compute_disparity_optimization
        disp_map, error_hist, iters = compute_disparity_optimization(
            left_image, right_image, block_size, max_disparity, method, num_pixels=200
        )

        runtime = time.time() - start_time
        results[method] = {
            'disparity': disp_map,
            'error_history': error_hist,
            'iterations': iters,
            'runtime': runtime
        }

        print(f"{method}: {runtime:.3f}s, {iters} total iterations")

    return results
