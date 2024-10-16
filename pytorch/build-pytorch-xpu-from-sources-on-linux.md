# Build PyTorch XPU from sources on Linux

## Supported hardware

Building GPU kernels for aten operators will be the most time and resource consuming operation when building PyTorch with XPU support. Prefer to use multi core/cpu system with enough RAM.

PyTorch XPU backend supports select Intel GPUs. That's documented by Intel per PyTorch release:

* [Here][prereq-2.5] for PyTorch 2.5
* [Here][prereq-2.4] for PyTorch 2.4

For details, see [Hardware supported by PyTorch XPU backend](hardware-supported-by-pytorch-xpu.md).

## Prerequisites installation

Prerequisites installation is documented by Intel per PyTorch release:

* [Here][prereq-2.5] for PyTorch 2.5
* [Here][prereq-2.4] for PyTorch 2.4

From these instructions, installation involves 2 steps:

1. Installation of Intel GPU driver packages following [separate documentation](https://dgpu-docs.intel.com/driver/installation.html#installation)
2. Installation of oneAPI packages with provided instruction

These steps will configure 2 separate package repositories which at the moment follow different architectures. Intel GPU drivers repository provides system wide installation of the packages (files got installed to `/usr`, `/etc`, etc.) with no further activation needed. Distinctly, oneAPI packages are installed as alternate packages to a special location (at `/opt/intel`) requiring special activation steps provided below as they appear in the documentation for PyTorch 2.5:
```
source /opt/intel/oneapi/pytorch-gpu-dev-0.5/oneapi-vars.sh
source /opt/intel/oneapi/pti/latest/env/vars.sh
```

It's required to install few prerequisite components to build and run PyTorch with XPU backend support:

* SYCL, Level Zero and OpenCL runtime development environments
* Intel oneAPI DPC++ Compiler, to build SYCL eager mode kernels
* Intel ocloc Compiler (included into Intel Compute Driver package) and Intel Graphics Compiler, to generate Intel GPU device specific code
* Intel Profiling Tools Interfaces (PTI) development environment

For Ubuntu 22.04 and PyTorch 2.5 these can be installed as follows:
```
sudo apt install -y \
  intel-level-zero-gpu \
  intel-opencl-icd \
  level-zero-dev
sudo apt install -y \
  intel-for-pytorch-gpu-dev-0.5 \
  intel-pti-dev
```

### Notes on GPU driver packages

Intel provides different set of packages for few use cases which are available in respective package repositories. First, packages differ depending on the GPU platform: one set is available for client platforms, another – for server platforms. Further, for server platforms packages are available in 2 flavors: as Long Term Support (LTS) packages and as rollowing stable packages. For client platforms only rolling stable packages are available. With rolling stable new features and optimizations are regularly available while LTS is more suitable for production with fewer updates limited to critical bug fixes. See https://dgpu-docs.intel.com/releases/index.html for more details. At the release of PyTorch 2.5 latest available LTS release of Intel driver packages was of `2350.103` version and rolling stable of `20240814` version.

Pay also attention that, depending on the Operating System and the GPU, kernel mode driver package might need to be installed. Typically that's the case for the newer GPUs support for which did not yet land in OS kernel.

For Ubuntu 22.04 mentioned package repositories can be configured as follows:

```
DGPU_KEY_FILE=/usr/share/keyrings/intel-graphics.gpg
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --yes --dearmor --output $DGPU_KEY_FILE

# for Server GPUs, 2350.x LTS
echo "deb [arch=amd64 signed-by=$DGPU_KEY_FILE] https://repositories.intel.com/gpu/ubuntu jammy/lts/2350 unified" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu.list

# alternatively, also for Server GPUs, but rolling stable:
echo "deb [arch=amd64 signed-by=$DGPU_KEY_FILE] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu.list

# for Client GPUs, rolling stable:
echo "deb [arch=amd64 signed-by=$DGPU_KEY_FILE] https://repositories.intel.com/gpu/ubuntu jammy client" | \
    sudo tee /etc/apt/sources.list.d/intel-gpu.list
```

### Notes on oneAPI

> [!CAUTION]
> Avoid mixing legacy and PyTorch specific oneAPI environments! This does not work!

Pay attention that oneAPI package repository for PyTorch 2.4 and 2.5 differs from legacy oneAPI repository which can be configured following [differrent install instruction](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&linux-install-type=apt). oneAPI PyTorch 2.5 package repository also provides different virtual top level package to install named `intel-for-pytorch-gpu-dev-0.5` instead of oneAPI legacy `intel-basekit`. Overall, for PyTorch 2.5 the following packages are recommended to be installed according to [documentation](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html):
```
sudo apt install intel-for-pytorch-gpu-dev-0.5 intel-pti-dev
```
As noted, the first package is a virtual package used to install multiple components it depends on. It's further possible to deduce required runtime and development packages if needed.

Usage of legacy oneAPI package repository to build and further run PyTorch 2.4 or 2.5 with XPU is not documented, but might be possible. This is however not tested and should be adviced against.

It's important to note that **mixing legacy and PyTorch specific oneAPI environments does not work**. PyTorch built in one environment won't run in another one. The following error was observed on such attempt:
```
undefined symbol: _ZN4sycl3_V15queue25ext_oneapi_submit_barrierERKSt6vectorINS0_5eventESaIS3_EERKNS0_6detail13code_locationE
```

Thus, it can be advised to isolate PyTorch XPU build and runtime environments by using dedicated system, virtual machine or docker container. Also, make sure to follow appropriate instructions, i.e. [these][prereq-2.5] for PyTorch 2.5:

```
ARG ONEAPI_KEY_FILE=/usr/share/keyrings/oneapi-archive-keyring.gpg
wget -qO - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
    gpg --yes --dearmor --output $ONEAPI_KEY_FILE

echo "deb [signed-by=$ONEAPI_KEY_FILE] https://apt.repos.intel.com/intel-for-pytorch-gpu-dev all main" | \
    tee /etc/apt/sources.list.d/oneapi.list
```

## Building PyTorch

Once prerequisite environment is setup and activated, PyTorch with XPU support can be built following PyTorch [build instruction](https://github.com/pytorch/pytorch/tree/release/2.5?tab=readme-ov-file#from-source) which can be outlined as follows:
```
pip install -r requirements.txt
python3 setup.py develop
USE_XPU=1 make triton
```
There are few things however worth to pay attention on.

### Notes on oneMKL

PyTorch build insruction recommends to install oneMKL library from conda using the following command:
```
conda install intel::mkl-static intel::mkl-include
```
Note that this step is optional and is used for optimization when PyTorch runs on CPU. If not using conda or building to run on Intel GPU, this step might be considered to be skipped.

### Notes on supported GPUs

By default eager mode aten operators are being pre-built for few specific Intel GPUs. For PyTorch 2.5 these are `pvc,xe-lpg,ats-m150` for Linux and `ats-m150,lnl-m,mtl-u,mtl-h` for Windows. If executed on an Intel GPU which is not in the pre-built list, runtime stack will attempt to recompile kernels on the fly. This operation is time consuming and might fail or lead to not functional kernel on a platform not supported by PyTorch XPU backend.

It is possible to adjust default GPUs list for eager mode kernels via `TORCH_XPU_ARCH_LIST` environment variable. This also can be useful to reduce compilation time. For example, the following command will build PyTorch with aten operators targeting only PVC:
```
TORCH_XPU_ARCH_LIST=pvc python3 setup.py develop
```

For details on this topic, see [Hardware supported by PyTorch XPU backend](hardware-supported-by-pytorch-xpu.md).

### Notes on Triton

The PyTorch 2.4 and 2.5 build command for Triton requires `USE_XPU=1` environment variable to build the Triton with XPU support:
```
USE_XPU=1 make triton
```
The reason behind is that XPU support is not available in upstream Triton and requires build from Intel [fork](https://github.com/intel/intel-xpu-backend-for-triton). So, `USE_XPU=1` environment variable swiches the build from upstream Triton to Intel fork.

In the case of active development in Triton itself, it might be useful to avoid building with PyTorch `USE_XPU=1 make triton` command and build in a cloned copy:
```
git clone https://github.com/intel/intel-xpu-backend-for-triton.git intel-triton && cd intel-triton
git reset --hard 91b14bf5593cf58a8541f3e6b9125600a867d4ef
pip install -e ./python
```
Where `91b14bf5593cf58a8541f3e6b9125600a867d4ef` commit is a commit from [.ci/docker/ci_commit_pins/triton-xpu.txt](https://github.com/pytorch/pytorch/blob/release/2.5/.ci/docker/ci_commit_pins/triton-xpu.txt) which sets validated commit from Triton fork repository for PyTorch 2.5 release.

### Notes on XPU support in PyTorch

When working on PyTorch source base, keep in mind that part of XPU code, specifically XPU implementation of aten operators, is available in a stand-alone Git repository. Namely https://github.com/intel/torch-xpu-ops. PyTorch build system hides this out performing a clone of the named repository during the build into `./third_party/torch-xpu-ops/` folder. The [third_party/xpu.txt](https://github.com/pytorch/pytorch/blob/release/2.5/third_party/xpu.txt) file provides a validated commit pin for this repo.

For any changes in torch-xpu-ops, create a local branch in the cloned copy and reconfigure `third_party/xpu.txt` file accordingly. For example:
```
cd third_party/torch-xpu-ops
git branch dev
cd ../../
echo "dev" > third_party/xpu.txt
```
Note that changes to torch-xpu-ops might be overwritten if `third_party/xpu.txt` is not adjusted accordingly.

## Known issues

See known build and version related issues below.

### PyTorch 2.5

* [intel/intel-xpu-backend-for-triton#2279]: No module named `triton.ops`

### PyTorch 2.4

* [pytorch/pytorch#130099]: xpu backend build fails if cloned directory is named python-x.y

[prereq-2.5]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html
[prereq-2.4]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-4.html

[pytorch/pytorch#130099]: https://github.com/pytorch/pytorch/issues/130999

[intel/intel-xpu-backend-for-triton#2279]: https://github.com/intel/intel-xpu-backend-for-triton/issues/2279
