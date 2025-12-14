import time
import numpy as np
import matplotlib.pyplot as plt
from load_images import load_stereo_images, rectify_stereo_pair, visualize_rectified_images
from compute_disparity import compute_disparity_block_matching, visualize_disparity_map, save_disparity_map
from optimize_disparity import optimize_disparity_levenberg_marquardt, plot_optimization_error, print_optimization_summary
from depth_reconstruction import disparity_to_depth, visualize_depth_map, create_depth_histogram, compare_depth_maps
from pointcloud import create_point_cloud, visualize_point_cloud, compute_point_cloud_statistics, save_point_cloud
from synthetic_data import generate_synthetic_stereo_pair, visualize_synthetic_data


def main(save_plots=True):
    """
    Main function for 3D Depth Estimation and Stereo Reconstruction.
    """
    print("=== 3D Depth Estimation and Stereo Reconstruction ===")
    print("Numerical Analysis Project")

    # Configuration
    BLOCK_SIZE = 15
    MAX_DISPARITY = 64
    NUM_OPTIMIZE_PIXELS = 500

    try:
        # 1. Load stereo images
        print("\n1. Loading stereo images...")
        left_image, right_image, camera_params = load_stereo_images(
            use_synthetic=True)

        print("Camera parameters:")
        for key, value in camera_params.items():
            print(f"  {key}: {value}")

        # Visualize synthetic data if generated
        if hasattr(left_image, 'shape'):  # Check if it's a numpy array
            _, _, gt_disparity, _ = generate_synthetic_stereo_pair()
            visualize_synthetic_data(left_image, right_image, gt_disparity,
                                     save_path="results/synthetic_data.png" if save_plots else None)

        # 2. Rectify stereo pair
        print("\n2. Rectifying stereo pair...")
        left_rectified, right_rectified, Q = rectify_stereo_pair(
            left_image, right_image, camera_params)
        visualize_rectified_images(left_rectified, right_rectified,
                                   save_path="results/rectified_images.png" if save_plots else None)

        # 3. Compute disparity using block matching
        print("\n3. Computing disparity (Block Matching)...")
        start_time = time.time()
        disparity_bm = compute_disparity_block_matching(left_rectified, right_rectified,
                                                        BLOCK_SIZE, MAX_DISPARITY)
        bm_time = time.time() - start_time
        print(".3f")

        visualize_disparity_map(disparity_bm, "Block Matching Disparity",
                                save_path="results/disparity_bm.png" if save_plots else None)
        save_disparity_map(
            disparity_bm, "results/disparity_block_matching.png")

        # 4. Compute disparity using numerical optimization
        print("\n4. Computing disparity (Levenberg-Marquardt Optimization)...")
        start_time = time.time()
        disparity_opt, error_history, convergence_messages, successful_opts = optimize_disparity_levenberg_marquardt(
            left_rectified, right_rectified, BLOCK_SIZE, MAX_DISPARITY, NUM_OPTIMIZE_PIXELS
        )
        opt_time = time.time() - start_time
        print(".3f")

        # Plot optimization error
        plot_optimization_error(error_history, "Levenberg-Marquardt Optimization Error",
                                save_path="results/optimization_error.png" if save_plots else None)

        # Print optimization summary
        print_optimization_summary(
            error_history, convergence_messages, opt_time, successful_opts)

        visualize_disparity_map(disparity_opt, "Optimized Disparity (Levenberg-Marquardt)",
                                save_path="results/disparity_opt.png" if save_plots else None)
        save_disparity_map(disparity_opt, "results/disparity_optimized.png")

        # 5. Depth reconstruction
        print("\n5. Converting disparity to depth...")

        # Block matching depth
        depth_bm = disparity_to_depth(
            disparity_bm, camera_params['focal_length'], camera_params['baseline'])
        visualize_depth_map(depth_bm, "Block Matching Depth Map",
                            save_path="results/depth_bm.png" if save_plots else None)

        # Optimized depth
        depth_opt = disparity_to_depth(
            disparity_opt, camera_params['focal_length'], camera_params['baseline'])
        visualize_depth_map(depth_opt, "Optimized Depth Map",
                            save_path="results/depth_opt.png" if save_plots else None)

        # Ground truth depth (if synthetic)
        if 'gt_disparity' in locals():
            depth_gt = disparity_to_depth(
                gt_disparity, camera_params['focal_length'], camera_params['baseline'])
            visualize_depth_map(depth_gt, "Ground Truth Depth Map",
                                save_path="results/depth_gt.png" if save_plots else None)

        # 6. Experimental analysis
        print("\n6. Experimental Analysis...")

        # Compare depth maps
        if 'depth_gt' in locals():
            metrics_bm, metrics_opt = compare_depth_maps(
                depth_bm, depth_opt, "Block Matching", "Optimized", depth_gt)
        else:
            compare_depth_maps(depth_bm, depth_opt,
                               "Block Matching", "Optimized")

        # Performance comparison
        print("\n=== Performance Comparison ===")
        print(f"Block Matching time: {bm_time:.3f} seconds")
        print(f"Optimization time: {opt_time:.3f} seconds")
        print(f"Speedup: {bm_time/opt_time:.2f}x")

        # Create performance table
        methods = ['Block Matching', 'Levenberg-Marquardt']
        times = [bm_time, opt_time]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(methods, times, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Computation Time Comparison')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_plots:
            plt.savefig("results/performance_comparison.png")
            plt.close()
        else:
            plt.show()

        # Disparity histograms
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(disparity_bm[disparity_bm > 0].flatten(
        ), bins=50, alpha=0.7, color='blue', label='Block Matching')
        plt.xlabel('Disparity (pixels)')
        plt.ylabel('Frequency')
        plt.title('Block Matching Disparity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(disparity_opt[disparity_opt > 0].flatten(
        ), bins=50, alpha=0.7, color='red', label='Optimized')
        plt.xlabel('Disparity (pixels)')
        plt.ylabel('Frequency')
        plt.title('Optimized Disparity Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_plots:
            plt.savefig("results/disparity_histograms.png")
            plt.close()
        else:
            plt.show()

        # 7. 3D Point Cloud Reconstruction
        print("\n7. Creating 3D point clouds...")

        # Block matching point cloud
        pc_bm = create_point_cloud(depth_bm, left_rectified, camera_params)
        compute_point_cloud_statistics(pc_bm)

        # Optimized point cloud
        pc_opt = create_point_cloud(depth_opt, left_rectified, camera_params)
        compute_point_cloud_statistics(pc_opt)

        # Save point clouds
        save_point_cloud(pc_bm, "results/pointcloud_block_matching.ply")
        save_point_cloud(pc_opt, "results/pointcloud_optimized.ply")

        # Visualize point clouds
        print("\nVisualizing Block Matching point cloud...")
        print("Close the plot window to continue to the optimized point cloud.")
        visualize_point_cloud(pc_bm, "Block Matching Point Cloud",
                              save_path="results/pointcloud_bm.png" if save_plots else None)

        print("\nVisualizing Optimized point cloud...")
        visualize_point_cloud(pc_opt, "Optimized Point Cloud",
                              save_path="results/pointcloud_opt.png" if save_plots else None)

        print("\n=== Analysis Complete ===")
        print("All outputs have been generated:")
        print("- Disparity maps saved as PNG files")
        print("- Point clouds saved as PLY files")
        print("- Visualizations displayed")

    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set save_plots=True to save plots to files instead of showing them
    main(save_plots=True)
