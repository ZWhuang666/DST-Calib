# User Guide

This guide describes the dataset layout and the training and evaluation entry
points. Complete the
[installation guide](Installation.md) before running these commands.

Run commands from the project root and activate the environment first:

```bash
conda activate dstcalib
```

## Dataset preparation

### Directory layout

A dataset root may contain one or more subsets, with each subset organized as
follows:

```text
<dataset_root>/
└── <subset_name>/
    ├── RGB/
    │   ├── <frame_id_0>.png
    │   ├── <frame_id_1>.png
    │   └── ...
    ├── pointclouds/
    │   ├── <frame_id_0>.pcd
    │   ├── <frame_id_1>.pcd
    │   └── ...
    ├── Depth/
    │   ├── <frame_id_0>.npz
    │   ├── <frame_id_1>.npz
    │   └── ...
    └── calib/
        └── <calibration_name>.json
```

The RGB image, point-cloud, and depth filenames must sort into the same frame
order. The folder names can be changed through `dataset_folder_format` in a
runtime configuration. The default mapping is:

```json
{
  "RGB_folder": "RGB",
  "PCloud_folder": "pointclouds",
  "Depth_folder": "Depth",
  "Calib_folder": "calib"
}
```

DST-Calib assumes that all input camera images have already been undistorted.
The current codebase does not provide an image-undistortion function, so users
must undistort their camera images before preparing the dataset.

### Calibration JSON

Each subset must contain at least one calibration JSON file. If there is only
one file, DST-Calib reuses it for every frame in that subset. If multiple files
are supplied, they must align with the sorted frame sequence.

The camera intrinsic matrix `K` and the ground-truth LiDAR-to-camera
extrinsics are required for training and quantitative evaluation. Distortion
coefficients may be included for compatibility with the full calibration
format. The rotation is represented as an OpenCV Rodrigues vector and the
translation as a three-dimensional vector:

```json
{
  "cloudCamera": {
    "Intrinsic": {
      "K": [
        [1000.0, 0.0, 640.0],
        [0.0, 1000.0, 360.0],
        [0.0, 0.0, 1.0]
      ],
      "D": [[0.0], [0.0], [0.0], [0.0], [0.0]]
    },
    "Extrinsic": {
      "R": [[0.0], [0.0], [0.0]],
      "T": [[0.0], [0.0], [0.0]]
    }
  }
}
```

See the [example calibration files](../example/TrueJson/) for the extended
format.

The `Depth` directory must be prepared before training or evaluation.

### Generate the Depth directory

Use `dataset_generate.py` to generate a depth directory for each matching
subset under a dataset root. For a one-level dataset layout, run:

```bash
python src/dataset_generate.py \
  --root-path /path/to/dataset_root \
  --key-word '*' \
  --depth-solution Moge2+DAR \
  --generate-method LIDAR_INI \
  --depth-folder-name Depth \
  --rank 1
```

The depth-generation options are:

- `--depth-solution`: `Moge2` generates monocular depth, while `Moge2+DAR`
  additionally rectifies it using the DAR;
- `--generate-method`: selects how the initial LiDAR-camera pose is chosen
  for depth rectification;
- `--depth-folder-name`: sets the output directory name. Use the same name in
  `dataset_folder_format.Depth_folder` when training or evaluating;
- `--rank`: use `1` when subsets are direct children of the dataset root, or
  `2` when they are nested under one additional grouping directory;
- `--key-word`: processes only subset names containing the specified text;
  `'*'` selects all subsets.

Three generation methods are available:

- `LIDAR_INI` uses the LiDAR origin pose as the initial pose. This is the
  recommended option when the initial pose error is small or its value is
  unknown. For example, this mode is suitable for KITTI;
- `INPUT_INI` uses a caller-provided initial transformation. This option is
  intended for programmatic use through `Dataset_batchDepthFolderGeneration`,
  because the command-line interface does not expose the transformation
  matrix;
- `AUTO_INI` automatically generates random initial poses within a
  neighborhood around the camera. Use it when the LiDAR and camera orientations
  differ substantially. For example, this mode is suitable for Argoverse,
  whose side and rear cameras do not share a field of view with the LiDAR's
  initial viewing direction.

For example, generate Argoverse depth files with automatic initialization:

```bash
python src/dataset_generate.py \
  --root-path /path/to/argoverse_root \
  --key-word '*' \
  --depth-solution Moge2+DAR \
  --generate-method AUTO_INI \
  --depth-folder-name Depth \
  --rank 1
```

For KITTI depth generation, we recommend using the following configuration for
more stable performance:

```bash
python src/dataset_generate.py \
  --root-path /path/to/KITTI_root \
  --key-word '*' \
  --depth-solution Moge2+DAR \
  --generate-method LIDAR_INI \
  --depth-folder-name Depth \
  --rank 1
```

The command skips a subset if the requested depth directory already exists.
Choose a new `--depth-folder-name`, or remove the existing directory only when
you intentionally want to regenerate its contents.

## Runtime configuration files

The official DST-Calib configurations are organized by workflow. They can be
used as references when preparing custom training and evaluation
configurations:

- [training configurations](../example/config/train/);
- [evaluation configurations](../example/config/test/).

Copy the appropriate file and update at least these paths and identifiers:

- `dataset_root`: root directory containing the selected subsets;
- `train_subset`, `val_subset`, or `test_subset`: subset directory names used
  by the selected entry point;
- `weight_folder` and `weight_name`: model checkpoint location;
- `dataset_folder_format`: folder-name mapping, if it differs from the default.

Subset names and sampling modes must be nested under
`select_dataset_mode_params`, with `train_method` set to `"select dataset"`.
The following excerpts show the required structure; retain the other fields
from the example configurations.

The training entry point loads `train_subset` and `val_subset`:

```json
{
  "train_method": "select dataset",
  "select_dataset_mode_params": {
    "train_subset": ["<train_subset>"],
    "train_sample_mode": [[1, 1, -1]],
    "val_subset": ["<validation_subset>"],
    "val_sample_mode": [[1, 1, -1]]
  }
}
```

The evaluation entry point only loads `test_subset`:

```json
{
  "train_method": "select dataset",
  "select_dataset_mode_params": {
    "test_subset": ["<test_subset>"],
    "test_sample_mode": [[1, 1, -1]]
  }
}
```

A sampling mode is written as `[start, gap, end]` with one-based indexing. For
example, `[1, 1, -1]` selects every frame from the first frame to the end.

Relative paths in JSON files are resolved from the process working directory.
The example commands therefore assume they are executed from the project root.

## Training

Create your own training configuration JSON using the
[example training configurations](../example/config/train/) as references,
then run the following command with the path to your configuration:

```bash
python src/train.py \
  --config /path/to/your/train_config.json \
  --test-ver DST_Calib
```

The training configuration defines the training and validation subsets,
sampling modes, batch size, epoch count, augmentation ranges, and output
checkpoint path. Ensure that `weight_folder` is writable.

To initialize training from an existing DST-Calib checkpoint and start a new
training cycle from epoch 0, set `pretrained` to `true` and point
`weight_folder` and `weight_name` to that checkpoint. This loads the model
weights without restoring the optimizer or scheduler state. For example, to
initialize training on Argoverse data from the KITTI-pretrained Base model,
adapt [train_Joint.json](../example/config/train/train_Joint.json) with:

```json
{
  "pretrained": true,
  "weight_folder": "./weights/Base",
  "weight_name": "DST_Base.pth"
}
```

Setting `pretrained` to `true` does not select or download a checkpoint.
Verify that the configured file exists before training: if it is missing,
the code prints `No checkpoint found` and starts from scratch.

To resume an interrupted training run, set `pretrained` to `false` (the
default) and point to the saved checkpoint. The code restores the model
weights, the optimizer and scheduler states when present, and resumes at
the epoch after the saved epoch.

The same `weight_folder` and `weight_name` fields also determine training
checkpoint output paths. Use a separate working copy of the downloaded
checkpoint when you want to preserve the original release files.

## Evaluation

Create your own evaluation configuration JSON using the
[example evaluation configurations](../example/config/test/) as references,
then run the following command with the path to your configuration and the
desired evaluation mode:

```bash
python src/test.py \
  --config /path/to/your/test_config.json \
  --test-ver DST_Calib \
  --mode SB
```

Supported modes are:

- `SB`: single-branch evaluation;
- `SB*`: joint evaluation using the SB pathway;
- `PE`: joint evaluation using the PE pathway;
- `PE+SB*`: joint evaluation using the combined pathway.

Quote mode names containing `*` when entering them in a shell, for example
`--mode 'SB*'`.
