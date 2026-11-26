# Installation

The recommended platform is Ubuntu 22.04 with an NVIDIA GPU and a compatible
driver. The instructions below create a Conda environment named `dstcalib`
with Python 3.10, PyTorch 2.3.1, and CUDA 11.8 support.

Run all commands from the project root unless stated otherwise.

## 1. Install Miniconda

Install [Miniconda](https://docs.anaconda.com/miniconda/), then open a new
terminal:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Recent Conda versions may ask you to accept the Anaconda Terms of Service:

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 2. Create the environment

```bash
conda create -n dstcalib python=3.10 -y
conda activate dstcalib
python -m pip install --upgrade pip
```

## 3. Install NumPy and PyTorch

Install the versions used by this project:

```bash
conda install numpy=2.0.1 -y

python -m pip install \
  torch==2.3.1 \
  torchvision==0.18.1 \
  torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

Verify that PyTorch can access the GPU:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

## 4. Install project dependencies

```bash
python -m pip install "setuptools<82" wheel
python -m pip install --no-build-isolation visdom==0.2.4
python -m pip install -r docs/requirements.txt
```

Confirm that the main numerical dependencies import correctly:

```bash
python - <<'PY'
import numpy as np
import scipy
import sklearn
import torch

print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
np.show_config()
PY
```

The reference environment uses NumPy 2.0.1 with MKL. If a later installation
step replaces NumPy or switches the BLAS backend, reinstall the pinned NumPy
package before running the experiments.

## 5. CUDA Chamfer Distance extension

The self-supervised loss can use the CUDA extension in
`src/losses/chamfer_distance`. It is substantially faster than the native
PyTorch fallback. Thanks to [otaheri](https://github.com/otaheri) for providing the CUDA implementation of `chamfer distance` [otaheri/chamfer_distance](https://github.com/otaheri/chamfer_distance).

PyTorch includes the CUDA runtime needed to execute CUDA operations, but
building this extension also requires `nvcc` and CUDA development headers. They
can be installed inside the Conda environment; a system-wide CUDA Toolkit is
not required. The host still needs a compatible NVIDIA driver and C++ compiler.

### Install the CUDA 11.8 build tools

The toolkit version should match `torch.version.cuda`. For the PyTorch build
installed above, use CUDA 11.8:

```bash
conda install \
  --override-channels \
  -c https://conda.anaconda.org/nvidia/label/cuda-11.8.0 \
  cuda-nvcc=11.8 \
  cuda-cudart-dev=11.8 \
  cuda-driver-dev=11.8

python -m pip install ninja==1.13.2
```

For a different PyTorch CUDA build, install the corresponding toolkit instead
of copying the CUDA 11.8 command unchanged.

### Verify the compiler

```bash
which nvcc
nvcc --version

python - <<'PY'
from torch.utils.cpp_extension import CUDA_HOME
print("CUDA_HOME:", CUDA_HOME)
PY
```

If `nvcc` is available but `CUDA_HOME` is `None`, set the environment paths:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
```

### Build and test the extension

The extension is compiled on its first import. From the project root, run:

```bash
PYTHONPATH=src \
python -c "import loss; print('CHAMFER_C:', loss.CHAMFER_C); print('Backend:', loss.ChamferDistance.__module__)"
```

A successful build reports `CHAMFER CUDA EXTENSION COMPILE SUCCESS` and
`CHAMFER_C: True`. Compiled extension inputs must be contiguous `float32`
tensors shaped `[B, N, 3]` on the CUDA device.

The backend is selected automatically. DST-Calib uses the CUDA extension when
CUDA, its build toolkit, and the extension sources are available. If these
requirements are not met, or if loading the extension fails, it falls back to
the native PyTorch implementation.

When reproducing the self-supervised experiments, confirm that the startup log
contains both `CHAMFER CUDA EXTENSION COMPILE SUCCESS` and `CHAMFER_C: True`.
If compilation fails, check that `torch.version.cuda` matches `nvcc --version`,
that a host C++ compiler is available, and that `CUDA_HOME` is correct.
