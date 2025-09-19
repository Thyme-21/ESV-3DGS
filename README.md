# ESGS
This is the official repository for our paper "Enhancing Sparse-View 3D Gaussian Splatting via Multi-Level Collaborative Optimization for Robust Novel View Synthesis"

<img width="1600" height="900" alt="1" src="https://github.com/user-attachments/assets/35557de9-1fe9-438f-85b2-ba4bd5143397" />


## Abstract
Novel view synthesis aims to reconstruct three-dimensional scenes from a limited number of viewpoints and generate photorealistic images of unseen views. This task has broad applications in virtual reality, film production, and robotic navigation. However, under sparse-view conditions, traditional methods such as 3D Gaussian Splatting (3DGS) often face challenges, including geometric instability and insufficient point cloud distribution.To address these issues, this paper introduces a multi-level collaborative optimization framework. Firstly, a monocular depth estimator provides depth-based geometric supervision, mitigating geometric deviations under sparse conditions. Secondly, a local refinement module based on cross-view matching dynamically corrects point cloud distribution in high-error regions, enhancing texture and detail consistency. Finally, a global-local collaborative point cloud density regulation method is introduced, combining global opacity control with depth-gradient awareness to compress redundant point clouds while preserving clear geometric boundaries.Experimental results on the LLFF and Mip-NeRF360 datasets demonstrate that our method achieves strong performance in geometric stability, detail fidelity, and overall rendering quality. Here, we show that our approach improves PSNR by up to 10%, SSIM by up to 8%, and reduces LPIPS by up to 15% compared to state-of-the-art methods, showcasing its potential for practical applications in sparse-view reconstruction.

## Installation
Tested on Ubuntu 20.04, CUDA 11.8, PyTorch 1.12.1

``````
conda env create --file environment.yml
conda activate ESGS
``````

``````
pip install gaussian-splatting/submodules/diff-gaussian-rasterization-confidence
pip install gaussian-splatting/submodules/simple-knn
``````

## Required Data
```
├── /data
   ├── mipnerf360
        ├── bicycle
        ├── bonsai
        ├── ...
   ├── nerf_llff_data
        ├── fern
        ├── flower
        ├── ...
```

## Evaluation

### LLFF

1. Download LLFF from [the official download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).

2. run colmap to obtain initial point clouds with limited viewpoints:
    ```bash
   python tools/colmap_llff.py
   ```

3. Start training and testing:

   ```bash
   # for example
   bash scripts/run_llff.sh ${gpu_id} data/nerf_llff_data/fern output/llff/fern
   ```

### MipNeRF-360

1. Download MipNeRF-360 from [the official download link](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).

2. run colmap to obtain initial point clouds with limited viewpoints:
    ```bash
   python tools/colmap_360.py
   ```

# Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [CoR-GS](https://github.com/jiaw-z/CoR-GS)
- [GS-LPM](https://github.com/Surrey-UP-Lab/GS-LPM)
