> [!WARNING]
> This package has been migrated to the [TeamTomo monorepo](https://github.com/teamtomo/teamtomo).
> Future development, bug fixes, and releases will happen there.
> This repository is archived and no longer maintained.
> This package is still published to and installable from the same PyPI project, but development installations should be made from the monorepo.

# torch-phase-shift

[![License](https://img.shields.io/pypi/l/torch-phase-shift.svg?color=green)](https://github.com/alisterburt/torch-phase-shift/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-phase-shift.svg?color=green)](https://pypi.org/project/torch-phase-shift)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-phase-shift.svg?color=green)](https://python.org)
[![CI](https://github.com/alisterburt/torch-phase-shift/actions/workflows/ci.yml/badge.svg)](https://github.com/alisterburt/torch-phase-shift/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alisterburt/torch-phase-shift/branch/main/graph/badge.svg)](https://codecov.io/gh/alisterburt/torch-phase-shift)

*torch-fourier-shift* is a package for shifting 1D, 2D and 3D images with subpixel precision 
by applying phase shifts to Fourier transforms in PyTorch.

<p align="center" width="100%">
  <img src="./docs/assets/shift_2d_image.png" alt="A 2D image and the shifted result" width="50%">
</p>

```python
import torch
from torch_fourier_shift import fourier_shift_image_2d

# create a dummy image
my_image = torch.tensor(
    [[0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0]]
)

# shift the image by 1 pixel in dim 0, 2 pixels in dim 1
shifted_image = fourier_shift_image_2d(image=my_image, shifts=torch.tensor([1, 2]))
```

API's are equivalent for 1D and 3D images.

## Installation

*torch-fourier-shift* is available on PyPI.

```shell
pip install torch-fourier-shift
```

## Usage

Please check the the docs at [teamtomo.org/torch-fourier-shift](https://teamtomo.org/torch-fourier-shift/)

### Caching

Some functions are equipped with an argument called cache_intermediates. If you set cache_intermediates=True, an LRU cache will be used to avoid recomputing intermediate results. Note that this might affect gradient calculations.

By default, the size of the cache is 3, and can be changed with an environmental variable called TORCH_FOURIER_SHIFT_CACHE_SIZE. Just do 
```
export TORCH_FOURIER_SHIFT_CACHE_SIZE=5
```

or


```
os.environ["TORCH_FOURIER_SHIFT_CACHE_SIZE"]=5
```
before importing the torch_fourier_shift module.

