# 3D Depth Estimation and Stereo Reconstruction Using Numerical Optimization

A complete Python implementation for 3D depth estimation from stereo image pairs using numerical optimization techniques.

## Project Overview

This project implements stereo vision algorithms for depth estimation, including:

- **Block Matching**: Traditional SSD-based disparity computation
- **Numerical Optimization**: Levenberg-Marquardt optimization for disparity refinement
- **Depth Reconstruction**: Converting disparity maps to 3D depth
- **Point Cloud Generation**: Creating 3D point clouds from depth data
- **Performance Analysis**: Comparing methods and visualizing results

## Features

- Synthetic stereo dataset generation with ground truth
- Modular implementation with separate components
- Comprehensive visualization and analysis
- Performance benchmarking
- 3D point cloud reconstruction

## Installation

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Note: Open3D is optional and may not be available for all Python versions. The code will work with matplotlib-based 3D visualization as a fallback.

## Usage

Run the complete pipeline:

```bash
python main.py
```

The script will:

1. Generate or load stereo images
2. Compute disparity using block matching
3. Refine disparity using numerical optimization
4. Convert disparity to depth maps
5. Generate 3D point clouds
6. Save all visualizations and results to the `results/` folder

## Output Files

The script generates the following outputs in the `results/` folder:

### Images

- `synthetic_data.png` - Input stereo pair and ground truth disparity
- `rectified_images.png` - Rectified stereo images
- `disparity_bm.png` - Block matching disparity map
- `disparity_opt.png` - Optimized disparity map
- `depth_bm.png` - Block matching depth map
- `depth_opt.png` - Optimized depth map
- `depth_gt.png` - Ground truth depth map

### Analysis

- `optimization_error.png` - Error convergence plot
- `performance_comparison.png` - Runtime comparison
- `disparity_histograms.png` - Disparity distributions

### 3D Data

- `pointcloud_block_matching.npz` - Block matching point cloud
- `pointcloud_optimized.npz` - Optimized point cloud
- `pointcloud_bm.png` - Block matching point cloud visualization
- `pointcloud_opt.png` - Optimized point cloud visualization

## Mathematical Background

### Disparity Computation

The disparity \(d\) between corresponding points in stereo images is found by minimizing the Sum of Squared Differences (SSD):

\[E(d) = \sum (I*{left}(x,y) - I*{right}(x-d,y))^2\]

### Depth from Disparity

Depth \(Z\) is calculated using the camera baseline \(B\) and focal length \(f\):

\[Z = \frac{f \cdot B}{d}\]

### 3D Point Reconstruction

3D coordinates are computed from pixel coordinates \((u,v)\) and depth \(Z\):

\[X = \frac{(u - c_x) \cdot Z}{f_x}\]
\[Y = \frac{(v - c_y) \cdot Z}{f_y}\]
\[Z = Z\]

## Project Structure

```
├── main.py                 # Main execution script
├── load_images.py          # Image loading and rectification
├── compute_disparity.py    # Block matching disparity
├── optimize_disparity.py   # Numerical optimization
├── depth_reconstruction.py # Depth map computation
├── pointcloud.py          # 3D point cloud generation
├── synthetic_data.py      # Synthetic dataset generation
└── requirements.txt       # Python dependencies
```

## Configuration

Key parameters in `main.py`:

- `BLOCK_SIZE`: Matching block size (default: 15)
- `MAX_DISPARITY`: Maximum disparity search range (default: 64)
- `NUM_OPTIMIZE_PIXELS`: Number of pixels to optimize (default: 500)

## Results

The implementation provides:

- Quantitative comparison between block matching and optimization methods
- Runtime performance analysis
- Depth accuracy metrics against ground truth
- Visual comparison of disparity and depth maps
- 3D point cloud visualization

## Dependencies

- numpy: Numerical computations
- scipy: Optimization algorithms
- opencv-python: Image processing
- matplotlib: Visualization
- open3d: 3D point cloud processing (optional)

## License

This project is for educational purposes in numerical analysis and computer vision.
