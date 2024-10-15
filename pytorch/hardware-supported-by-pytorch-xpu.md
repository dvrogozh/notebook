# Hardware supported by PyTorch XPU backend

## Preface

XPU backend is available in PyTorch starting from PyTorch 2.4 and requires Intel GPU to run. However, not all Intel GPUs are supported. This article reviews hardware support story for PyTorch XPU backend from different angles. We will review what hardware is supported by Intel and duscuss important implementation details affecting hardware support.

## Intel GPUs supported for PyTorch XPU backend by Intel

Intel GPUs supported for PyTorch XPU backend are documented by Intel per PyTorch release:

* [Here][prereq-2.5] for PyTorch 2.5
* [Here][prereq-2.4] for PyTorch 2.4

Not all GPUs are declared for support by all Operating Systems. Tables below provide summary:

* For Linux (Ubuntu, RHEL, SUSE) – Server GPUs are supported (starting from PyTorch 2.4):

|                                     | `2.4`      | `2.5`      |
| ----------------------------------- | :--------: | :--------: |
| [Intel® Data Center GPU Max Series] | &#x2713;   | &#x2713;   |

* For Windows (10 and 11, including WSL2) – Client GPUs are supported (starting from PyTorch 2.5):

|                                     | `2.5`      |
| ----------------------------------- | :--------: |
| **Discrete Client GPUs:**           |            |
| [Intel® Arc™ A-Series Graphics]     | &#x2713;   |
| **Integrated Client GPUs:**         |            |
| [Meteor Lake]                       | &#x2713;   |
| [Lunar Lake]                        | &#x2713;   |

## Intel GPUs for which PyTorch XPU has pre-built eager mode kernels

PyTorch XPU implements eager mode Aten operators in a few different ways:

* Most operators are implemented as SYCL kernels
* Convolution and GEMM operators are implemented via [oneDNN] library

[oneDNN] supports wide range of Intel GPUs. Check with its documentaiton for details (roughly - starting from Tiger Lake). SYCL kernels however are generated (by default) for the smaller range of select GPUs which is also made OS dependent. The list of GPUs can be found:

* For Linux:

  * [Here](https://github.com/intel/torch-xpu-ops/blob/7e3d00acea9f0d3728048a5b2743de20d55c64ba/cmake/BuildFlags.cmake#L122) for PyTorch 2.5
  * [Here](https://github.com/intel/torch-xpu-ops/blob/97d692eb8c4b3afab17700a2fd918adcea0cba45/cmake/BuildFlags.cmake#L71) for PyTorch 2.4

* For Windows:

  * [Here](https://github.com/intel/torch-xpu-ops/blob/7e3d00acea9f0d3728048a5b2743de20d55c64ba/cmake/BuildFlags.cmake#L120) for PyTorch 2.5

Below tables provide summary on the device types for which SYCL kernels are pre-built in PyTorch XPU:

* For Linux:

| Device Type   | `2.4`      | `2.5`      |
| ------------- | :--------: | :--------: |
| `ats-m150`    |            | &#x2713;   |
| `pvc`         | &#x2713;   | &#x2713;   |
| `xe-lpg`      | &#x2713;   | &#x2713;   |

* For Windows:

| Device Type   | `2.5`      |
| ------------- | :--------: |
| `ats-m150`    | &#x2713;   |
| `mtl-h`       | &#x2713;   |
| `mtl-u`       | &#x2713;   |
| `lnl-m`       | &#x2713;   |

Device types in the tables above are given as they appear in the PyTorch XPU sources. They correspond to device types accepted by `ocloc` compiler with `-device <device_type>` argument. Full list of accepted types depends on the installed version of Intel GPU drivers stack and can be queried by running `ocloc compile --help`.

Note that it's possible to adjust list of devices for which SYCL kernels will be compiled during PyTorch XPU build process. For example, the following build command line corresponds to default PyTorch 2.5 XPU build setting on Linux:
```
TORCH_XPU_ARCH_LIST=pvc,xe-lpg,ats-m150 python3 setup.py develop
```

Building for a lesser number of GPUs will reduce build time and footprint. This also can be used to attempt a build for a different Intel GPUs. Note however that such a build might not succeed or generated binaries might fail at runtime.

## Eager mode kernels recompilation at runtime

One important aspect of how XPU backend in PyTorch works is potential kernels recompilation at runtime. It is triggered by underlying driver stack if pre-built eager mode kenrels are not available for the current GPU. Such recompilation is time consuming and will cause higher latency. It also might fail at compile time or further during kernel execution on device if kernel implementation is incompatible with this particular GPU.

At the moment PyTorch XPU has no built-in checks to verify current GPU against supported list (see [pytorch/pytorch#131799]). Thus, if underlying driver stack will report that it's available for the GPU, then PyTorch XPU will try to use such device potentially causing runtime kernels recompilation.

[prereq-2.5]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-5.html
[prereq-2.4]: https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-4.html

[Intel® Arc™ A-Series Graphics]: https://ark.intel.com/content/www/us/en/ark/products/series/227957/intel-arc-a-series-graphics.html
[Intel® Data Center GPU Flex Series]: https://ark.intel.com/content/www/us/en/ark/products/series/230021/intel-data-center-gpu-flex-series.html
[Intel® Data Center GPU Max Series]: https://ark.intel.com/content/www/us/en/ark/products/series/232874/intel-data-center-gpu-max-series.html

[Lunar Lake]: https://ark.intel.com/content/www/us/en/ark/products/codename/213792/products-formerly-lunar-lake.html
[Meteor Lake]: https://ark.intel.com/content/www/us/en/ark/products/codename/90353/products-formerly-meteor-lake.html

[oneDNN]: https://github.com/oneapi-src/oneDNN

[pytorch/pytorch#131799]: https://github.com/pytorch/pytorch/issues/131799
