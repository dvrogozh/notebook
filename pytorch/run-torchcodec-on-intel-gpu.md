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

Note that TorchCodec is under active development and [PR-558] might be frequently rebased. Sometimes build of TorchCodec against arbitrary version of FFmpeg might fail. Typically this happens either if TorchCodec does not yet support selected newer version of FFmpeg (for example, from master) or if FFmpeg throws deprecation warnings on the APIs used in TorchCodec. If this happens, try to adjust selected FFmpeg version or patch TorchCodec to bypass the issue. Below are FFmpeg versions used to verify [PR-558] in respect to TorchCodec base commits:

| TorchCodec | FFmpeg   |
| ---------- | -------- |
| [b01942c]  | `n6.1.2` |

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

If running under Docker, make sure to map `-v /dev/dri/by-path:/dev/dri/by-path` as this file tree is used by [PR-558] to query PyTorch devices to `/dev/dri/` device files.

In addition to the tests the following script can be used to dump rough decoded streams in RGB format:

```
import argparse
import numpy
import torch

from torchcodec.decoders import VideoDecoder

parser = argparse.ArgumentParser(description="TorchCodec decoder script")

def _parse_slice(s):
    ss = [int(e) if e.strip() else None for e in s.split(":")]
    return slice(*ss)

parser.add_argument("--input", "-i", type=str, required=True, help="Stream to decode")
parser.add_argument("--output", "-o", type=str, required=True, help="Name of output YUV/RGB file")
parser.add_argument("--device", "-d", type=str, default="cpu", help="PyTorch device to use (cpu, cuda, cuda:0, etc.)")
parser.add_argument("--slice", "-s", default=slice(None), type=_parse_slice)

args = parser.parse_args()

def dump_rgb(filename, tensor):
    tensor_numpy = torch.permute(tensor, (0,2,3,1)).numpy()
    tensor_numpy.tofile(filename)

decoder = VideoDecoder(args.input, device=args.device)
print(decoder.metadata)
dump_rgb(args.output, decoder[args.slice].cpu())
```

For example, on the test mp4 clip from TorchCodec repository:

```
python3 ../dump2.py -i test/resources/nasa_13013.mp4 -o out.rgb -d xpu -s 0:10
```

[TorchCodec]: https://github.com/pytorch/torchcodec.git
[PR-558]: https://github.com/pytorch/torchcodec/pull/558
[b01942c]: https://github.com/pytorch/torchcodec/commit/b01942c677c7660b54a925cccbe927a78c783287
