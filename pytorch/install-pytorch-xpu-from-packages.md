# Install PyTorch XPU from binary packages

PyTorch XPU binary packages became available at PyTorch 2.5 timeframe. Nightly and test preview PyPI repositories were provided first. Release repository is pending on [pytorch/pytorch#135867].

## Supported hardware

PyTorch XPU backend supports select Intel GPUs. That's documented by Intel per PyTorch release:

* [Here][prereq-2.5] for PyTorch 2.5

For detailed review, see [Hardware supported by PyTorch XPU backend](hardware-supported-by-pytorch-xpu.md).

## Available repositories

PyTorch XPU binary packages became available at PyTorch 2.5 timeframe. At the moment nightly and test preview PyPI repositories are provided.

> [!NOTE]
> Release repository is not yet available due to [pytorch/pytorch#135867].

Test preview version of PyTorch with XPU backend can be installed with the following command:
```
pip3 install torch --index-url https://download.pytorch.org/whl/test/xpu
```

Nightly version can be installed with:
```
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/xpu
```

## Prerequisite packages

`torch` package installed from one of the previously described PyPI repositories includes minimal set of oneAPI runtime dependencies to make it functional. However, these dependencies alone are not enough to offload execution to Intel GPU. Effectively, after installing `torch` user should be able to import PyTorch and query whether XPU support is available (which might return `False` or `True` depending on whether Intel GPU driver packages were installed or not). Running PyTorch on CPU should also be available.

So, this might be an outcome if no Intel GPU driver packages were installed:
```
# python3 -c 'import torch; print(torch.xpu.is_available())'
False
```

At the moment, with PyTorch 2.5, to make PyTorch XPU fully functional (including eager mode, `torch.compile`, profiling, etc.) it's required to install full set of prerequisite packages outlined by Intel:

* [Here][prereq-2.5] for PyTorch 2.5

Below we will discuss which components are required for some specific PyTorch XPU features. For general considerations on available install options, refer to [Build PyTorch XPU from sources on Linux](build-pytorch-xpu-from-sources-on-linux.md) review.

### Packages for PyTorch XPU eager mode

The following components are required to make PyTorch XPU eager mode functional:

* SYCL runtime, Level Zero runtime and Level Zero driver are required for overall GPU runtime support and kernels loading
* Intel OpenCL driver is required for Convolution and GEMM support, that's [oneDNN] library dependency

These components can be installed as follows on Ubuntu 22.04:

```
sudo apt install -y \
  intel-level-zero-gpu \
  intel-opencl-icd \
  level-zero
```

The following tests can be executed for the quick check:
```
# python3 -c 'import torch; print(torch.xpu.is_available())'
True

# python3 -c "import torch; \
    print(torch.zeros([2, 4], dtype=torch.int32, device='xpu'))"
tensor([[0, 0, 0, 0],
        [0, 0, 0, 0]], device='xpu:0', dtype=torch.int32)

# python3 -c "import torch; \
    t = torch.randn(20, 16, 50).to('xpu'); \
    print(torch.nn.Conv1d(16, 33, 3, stride=2).to('xpu')(t))"
tensor([[[-1.1783e-01,  3.1633e-01, -1.0514e+00,  ..., -3.9691e-01,
           7.1970e-02,  4.7188e-01],
       ...,
         [-2.8428e-01, -1.7172e-02,  1.2454e-01,  ..., -5.4396e-01,
           6.0884e-01, -1.1972e-01]]], device='xpu:0',
       grad_fn=<ConvolutionBackward0>)
```

### Packages for PyTorch compile mode

PyTorch compile mode uses Triton to compile kernels. With PyTorch 2.5 only Intel DPC++ compiler is supported without dedicated efforts to customize environment or patch Triton scripts to make use of clang or gcc (see [pytorch/pytorch#137518] and [intel/intel-xpu-backend-for-triton#2441]).

Altogether, the following components are needed to make PyTorch compile mode work on XPU backend:

* Intel DPC++ compiler
* SYCL runtime development environment
* Level Zero runtime and Level Zero driver

> [!NOTE]
> To use clang or gcc it's required to have SYCL runtime development environment installed. At the moment it's getting installed with Intel DPC++ compiler. Note that Triton version included into PyTorch 2.5 also makes preference to DPC++ compiler and requires compiler with default enabled `c++17` support (`clang-16` or later). These would be changed needed to use clang or gcc with Intel version of Triton.

These components can be installed with the following commands on Ubuntu 22.04:

```
sudo apt install -y \
  intel-level-zero-gpu \
  level-zero
sudo apt install -y \
  intel-oneapi-compiler-dpcpp-cpp-2024.1
```

Make sure to active oneAPI runtime environment to get access to DPC++ compiler:

```
source /opt/intel/oneapi/setvars.sh
```

The following test can be executed for the quick check:

```
# cat test.py
import torch

def foo(x, y):
    a = torch.sin(x)
    b = torch.cos(y)
    return a + b

foocomp = torch.compile(foo)
t = foocomp(torch.randn(10, 10).to('xpu'), torch.randn(10, 10).to('xpu'))
print('torch compile sin+cos:', t)

# python3 test.py
torch compile sin+cos: tensor([[ 0.5134,  1.8229,  1.0994,  0.1453,  0.9209,  1.9050,  0.7602,  1.2677,
          0.0341,  0.3029],
        [-0.3732, -0.3783, -0.0369,  0.0577,  0.6406, -0.8880, -1.2099,  0.4080,
          0.9581,  0.5883],
        [ 0.1470,  1.3212,  1.0451,  1.8429,  0.9579,  1.9803,  1.1809,  0.7117,
          0.2369,  0.6678],
        [-0.0383,  1.7047,  1.6223,  0.8934,  0.3135, -0.3435, -0.7599,  0.3086,
          1.3828,  0.5831],
        [ 0.6800,  0.4719,  1.5033, -0.6452, -1.2774,  1.8553,  1.6811,  0.5702,
          0.9438,  0.2748],
        [ 0.1880,  1.2848,  0.9971,  1.3123,  0.7295, -1.1626, -1.3340,  1.7450,
          0.7555,  1.1958],
        [ 0.5357, -0.8779, -0.1700,  1.2363, -0.4741,  0.3358,  0.3161,  1.5532,
          1.9477,  1.1257],
        [-0.0066,  0.0779,  1.2375,  0.4681,  1.3629, -0.6930,  0.0743,  0.7985,
          1.0244,  1.0434],
        [-0.0514,  1.3612,  1.1562, -0.9856,  0.3025,  0.8627,  0.0401, -0.7299,
          1.6957, -0.3850],
        [-0.3873,  0.5930,  0.5662,  0.6198,  0.7924,  0.3271,  1.9531,  1.8429,
          0.3158,  1.7810]], device='xpu:0')
```

[prereq-2.5]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html
[prereq-2.4]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-4.html

[oneDNN]: https://github.com/oneapi-src/oneDNN

[pytorch/pytorch#135867]: https://github.com/pytorch/pytorch/issues/135867
[pytorch/pytorch#137518]: https://github.com/pytorch/pytorch/issues/137518

[intel/intel-xpu-backend-for-triton#2441]: https://github.com/intel/intel-xpu-backend-for-triton/issues/2441
