# GL-GS
This is the official repository for our paper "A Global-Local Consistency Framework for Sparse-view 3D Gaussian Splatting"

![image](https://github.com/user-attachments/assets/bd7041f8-285b-488b-a4c9-e610b054615b)

## Abstract
Substantial progress has been made in novel view synthesis in recent years. However, achieving high-quality 3D reconstruction under sparse-view conditions remains challenging. Traditional methods still face the difficulties with geometric distortions and loss of local details when only a limited number of images are available as input. In this paper, a new framework, GL-GS, which integrates both global and local information, is proposed to improve viewpoint synthesis in the domain of sparse reconstruction. Initially, global geometric constraints, established through a dual-path Gaussian radiance field, are employed to identify potentially flawed regions and perform preliminary optimization of the overall geometric structure. Building upon this, a local optimization mechanism based on error maps and weakly supervised region matching is introduced to further pinpoint unstable regions. This mechanism applies strategies for point densification and dynamically adjusts transparency to precisely correct local geometric inaccuracies. Lastly, to better ensure the perceptual consistency in local details between the rendered image and the real image,we design a patch-based ranking loss. By randomly sampling local patches and imposing constraints on the relative ordering of pixel pairs, this loss enhances the perceptual quality of the reconstructed images. Experimental results demonstrate that our method provides a practical global-local fusion approach for 3D scene reconstruction under sparse-view conditions and achieves superior rendering quality across multiple datasets.

## Installation
Tested on Ubuntu 20.04, CUDA 11.8, PyTorch 1.12.1

``````
conda env create --file environment.yml
conda activate GLGS
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
