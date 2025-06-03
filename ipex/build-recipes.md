# Build recipes

[IPEX] provides 2 major features within PyTorch ecosystem:
* Implementations of operations for Intel GPUs missing in stock PyTorch (here IPEX can be considered as an out-of-tree PyTorch plugin extension)
* Implementations of operations popular within ecosystem projects, but not implemented in PyTorch

## IPEX for Intel GPUs

Support for Intel GPUs lives in [xpu-main] branch of [IPEX] repository and requires particular PyTorch versions to be installed. To be on the safe side, always check [dependency_version.json] file for the compatible versions of the key components including pytorch.

### Building 9d114fc5d (May 2025)

IPEX of [9d114fc5d] requires oneAPI 2025.1 components to be installed. On Ubuntu install them as follows:

```
wget -qO- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | \
  sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt-get update
apt-get install intel-deep-learning-essentials-2025.1
```

Next, install compatible version of PyTorch and setup oneAPI environment variables:

```
pip3 install --pre \
  torch==2.8.0.dev20250521+xpu \
  torchaudio==2.6.0.dev20250521+xpu \
  torchvision==0.22.0.dev20250521+xpu \
  --index-url https://download.pytorch.org/whl/nightly/xpu

. /opt/intel/oneapi/compiler/2025.1/env/vars.sh
. /opt/intel/oneapi/umf/0.10/env/vars.sh
. /opt/intel/oneapi/pti/0.12/env/vars.sh
. /opt/intel/oneapi/ccl/2021.15//env/vars.sh
. /opt/intel/oneapi/mkl/2025.1/env/vars.sh
```

With the above setup it should be possible to build IPEX for Intel GPUs as follows:

```
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex && cd ipex
git checkout 9d114fc5d  # from xpu-main branch
git submodule update --init --recursive

pip3 install -r requirements.txt
TORCH_XPU_ARCH_LIST=pvc python3 setup.py develop
```

Note that `TORCH_XPU_ARCH_LIST=pvc` is a developer specific environment variable to limit set of architectures to compile SYCL kernels for. This speeds up building IPEX.

[IPEX]: https://github.com/intel/intel-extension-for-pytorch
[xpu-main]: https://github.com/intel/intel-extension-for-pytorch/tree/xpu-main
[dependency_version.json]: https://github.com/intel/intel-extension-for-pytorch/blob/xpu-main/dependency_version.json

[9d114fc5d]: https://github.com/intel/intel-extension-for-pytorch/commit/9d114fc5d34db2c54874d3ed5dfd1dd6a944feff
