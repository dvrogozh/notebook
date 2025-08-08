# Run TorchCodec in Intel GPU

This document goes thru steps to patch [TorchCodec] to run it on Intel GPUs with PyTorch XPU backend. The patch to apply can be found in the following PR:

* https://github.com/pytorch/torchcodec/pull/558

Note that usage of TorchCodec requres Intel GPU supported by PyTorch and having media decoding/encoding capabilities such as Intel BattleImage or Alchemist GPUs. Intel PonteVecchio GPU can't be used as it does not have media decoding/encoding engines. Below we assume that appropriate system is being used and it was configured for both media and compute usages according to https://dgpu-docs.intel.com/driver/overview/overview.html.

Install PyTorch with enabled XPU backend:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/xpu
```

Install Intel oneAPI Deep Learning Essentials which matches pytorch version which got installed. For example, for PyTorch 2.7 follow https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-7.html. This step is required to be able to compile TorchCodec from sources. Otherwise the following error can be observed:
```
CMake Error at /home/dvrogozh/pytorch.xpu/lib/python3.12/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake:61 (set_target_properties):
  The link interface of target "c10_xpu" contains:

    torch::xpurt

  but the target was not found.  Possible reasons include:

    * There is a typo in the target name.
    * A find_package call is missing for an IMPORTED target.
    * An ALIAS target is missing.

```

Install build and runtime prerequisites:
```
sudo apt-get install -y \
  libmp3lame-dev \
  libva-dev \
  libze-dev \
  nasm \
  pybind11-dev

pip3 install \
  numpy \
  pytest \
  torcheval
```

Setup environment variables:
```
. /opt/intel/oneapi/setvars.sh
export PKG_CONFIG_PATH=~/_install/lib/pkgconfig/:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=~/_install/lib/:$LD_LIBRARY_PATH
export PATH=~/_install/bin/:$PATH
```

Build and install ffmpeg:

```
git clone https://github.com/FFmpeg/FFmpeg.git ffmpeg && cd ffmpeg
git checkout n7.1.1
./configure \
  --prefix=$HOME/_install \
  --libdir=$HOME/_install/lib \
  --disable-static \
  --disable-doc \
  --enable-shared \
  --enable-vaapi \
  --enable-libmp3lame
make -j$(nproc)
make install
```

Fetch, patch and install TorchCodec:
```
git clone https://github.com/pytorch/torchcodec.git && cd torchcodec
git fetch origin pull/558/head:pr_558
git checkout pr_558
git branch -u origin/main

ENABLE_XPU=1 python3 setup.py develop
```

On successful setup it should be possible to execute TorchCodec tests. Most should pass with the exception of few:
```
python3 -m pytest test/
```

[TorchCodec]: https://github.com/pytorch/torchcodec.git
