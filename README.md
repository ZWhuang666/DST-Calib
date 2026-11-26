# DST-Calib

DST-Calib is a target-free network for LiDAR-camera extrinsic calibration. It
uses double-sided data augmentation to improve generalization across scenes and
sensor configurations, and supports both fully supervised and self-supervised
learning.

## Publication

Our paper, **"DST-Calib: A Dual-Path, Self-Supervised, Target-Free
LiDAR-Camera Extrinsic Calibration Network,"** was published in *IEEE
Transactions on Robotics* (T-RO).

- [Paper on arXiv](https://arxiv.org/abs/2601.01188)
- [IEEE DOI](https://doi.org/10.1109/TRO.2026.3741440)

If you find this project useful, please cite:

```bibtex
@article{huang2026dstcalib,
  title   = {DST-Calib: A Dual-Path, Self-Supervised, Target-Free LiDAR-Camera Extrinsic Calibration Network},
  author  = {Huang, Zhiwei and Fu, Yanwei and Zhou, Yi and Chen, Xieyuanli and Chen, Qijun and Fan, Rui},
  journal = {IEEE Transactions on Robotics},
  year    = {2026},
  doi     = {10.1109/TRO.2026.3741440}
}
```

## Dataset and Pretrained Weights

The LCScenes dataset, DST-Calib pretrained weights, and MoGe-2 depth
estimation model are shared together through the links below:

- [Google Drive](https://drive.google.com/drive/folders/16X5m_q2_pMS040Vsa1eCcNRmqhbg4-4W?usp=sharing)
- [Baidu Netdisk](https://pan.baidu.com/s/1NGyDfTIXksLXsAxx8ROolA?pwd=h8xc)

Each shared root folder in the links contains `LCScenes/` and `weights/`, organized as
described below.

### LCScenes Dataset

LCScenes contains **10 subsets**, named `ZV01` through `ZV10`. The dataset
download is organized as follows. Each subset has the same six folders:

```text
LCScenes/
|-- readme.md
|-- ZV01/
|   |-- calib/
|   |-- Depth/
|   |-- pointclouds_bin/
|   |-- pointclouds/
|   |-- RGB/
|   `-- RGB_origin/
|-- ZV02/
|-- ...
`-- ZV10/
```

| Folder | Description |
| --- | --- |
| `calib` | Ground-truth calibration files. |
| `Depth` | Camera depth estimates that have been generated and rectified. |
| `pointclouds_bin` | Raw point clouds captured by the LiDAR sensor. |
| `pointclouds` | Downsampled point clouds with smaller file sizes, suitable for network inference and training. |
| `RGB` | Undistorted camera images. |
| `RGB_origin` | Original camera images before undistortion. |

Folder names are case-sensitive. See the [user guide](docs/UserGuide.md) for
dataset configuration and usage.

### Pretrained Weights

The weights download is organized as follows:

```text
weights/
|-- Base/
|   |-- DST_Base.pth
|   `-- readme.txt
|-- Joint/
|   |-- DST_Joint.pth
|   `-- readme.txt
`-- Moge2/
    `-- model.pt
```

| Checkpoint | Training Data |
| --- | --- |
| `Base/DST_Base.pth` | KITTI Odometry sequence 08 (left camera). |
| `Joint/DST_Joint.pth` | KITTI Odometry sequence 08 and Argoverse 2 sequences 000_00--000_01, initialized from the KITTI-pretrained Base model. |

Both DST-Calib models are trained under a double-sided data augmentation range 2*[5°, 0.5m] (using axis-wise weights of [0.6, 0.2, 0.2] applied to axes of the range). After downloading, set `weight_folder` and `weight_name` in your runtime configuration to the checkpoint folder and filename, respectively. See the [user guide](docs/UserGuide.md) for training and evaluation instructions.

### MoGe-2 Depth Estimation Model

The MoGe-2 checkpoint was downloaded from the official model page and is
included in our shared download folders for convenience. Credit for this
model belongs to the MoGe authors.

MoGe-2 is used to generate monocular camera depth estimates during dataset
preparation. The `Moge2+DAR` workflow additionally rectifies these estimates
using DAR. Download the model from `weights/Moge2/` in either shared root folder above and
place it at the following path relative to the project root:

```text
weights/Moge2/model.pt
```

When commands are run from the project root, the depth-generation code
automatically loads this local checkpoint if it exists. Otherwise, it loads
`Ruicheng/moge-2-vitl-normal` from Hugging Face, which may require an internet
connection. The MoGe-2 checkpoint is loaded separately from the DST-Calib
checkpoints and does not use `weight_folder` or `weight_name`.

LCScenes already includes generated and rectified depth estimates in its
`Depth` folders. To generate depth estimates for other datasets, follow the
[depth-generation instructions](docs/UserGuide.md#generate-the-depth-directory).

## Installation

See the [installation guide](docs/Installation.md) for environment setup,
dependencies, and optional CUDA Chamfer Distance extension instructions.

## Getting Started

See the [user guide](docs/UserGuide.md) for data preparation, training,
and evaluation workflows.

## Acknowledgements

This project builds upon components and ideas from
[CalibNet](https://github.com/epiception/CalibNet) and
[MoGe](https://github.com/microsoft/MoGe). We thank their authors for making
their work publicly available.
