# Known limitations

## PyTorch 2.5

* [intel/intel-xpu-backend-for-triton#2279]: No module named `triton.ops`
* [pytorch/pytorch#135766]: xpu: torchaudio build fails with torch::xpurt target not found with cmake<3.25

  * Apply [pytorch/pytorch#135767] for the fix

### PyTorch 2.4

* [pytorch/pytorch#126488]: xpu: provide a way to debug explicit CPU fallback

  * Apply [intel/torch-xpu-ops#318] to https://github.com/intel/torch-xpu-ops for the fix

* [pytorch/pytorch#130099]: xpu backend build fails if cloned directory is named python-x.y


[pytorch/pytorch#126488]: https://github.com/pytorch/pytorch/issues/126488
[pytorch/pytorch#130099]: https://github.com/pytorch/pytorch/issues/130999
[pytorch/pytorch#135766]: https://github.com/pytorch/pytorch/issues/135766

[pytorch/pytorch#135767]: https://github.com/pytorch/pytorch/pull/135767

[intel/torch-xpu-ops#318]: https://github.com/intel/torch-xpu-ops/pull/318

[intel/intel-xpu-backend-for-triton#2279]: https://github.com/intel/intel-xpu-backend-for-triton/issues/2279
