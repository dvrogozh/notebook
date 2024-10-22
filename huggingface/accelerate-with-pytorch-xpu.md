# Running Accelerate with PyTorch XPU backend

# Availability

Haggingface Accelerate supports PyTorch XPU backend starting from version `0.32.0` (see [huggingface/accelerate#2825]). Previously, support for Intel Extension for PyTorch (IPEX) was available and got extended to cover direct integration of XPU backend in PyTorch (available starting from PyTorch 2.4).

# Testing

Accelerate library can be formally tested for XPU backend by the project included tests on the system with supported Intel GPU graphics:

```
cd /path/to/accelerate/clone/copy
python3 -m pytest tests
```

# Using with PyTorch 2.5

| Accelerate | Failed | Passed | Skipped |
| --- | --- | --- | --- |
| `v0.34.2` | 19 | 214 | 142 |

PyTorch 2.5 includes a fix for [pytorch/pytorch#127929] which unlocked Accelerate tests. On `v0.34.2` passing rate is 91.8% with 19 tests failed, 214 passed and 142 skipped (executed on a system with a single [Intel® Data Center GPU Max 1100] card). The most significant issue affecting majority of tests (16 out of 19) is [pytorch/pytorch#135550] on PyTorch side:

* [pytorch/pytorch#135550]: xpu: runtime error in safetensors: Error while trying to find names to remove to save state dict

2 changes are needed to address this issue, one on PyTorch side – [pytorch/pytorch#135567], another in Triton – [intel/intel-xpu-backend-for-triton#2192] (see [pytorch/pytorch#137886] for Triton commit pin update in PyTorch):

Skipped tests require multiple GPUs, XPU enabled `bitsandbytes` library and/or other missing libraries. This needs further attention and enabling efforts.

# Using with PyTorch 2.4

While support of PyTorch XPU backend was initially enabled in Accelerate against PyTorch 2.4, it was limited to few manual tests executed with Huggingface Transformers (see [huggingface/transformers#31237] for details). Thoroughly testing Accelerate using project provided [tests](https://github.com/huggingface/accelerate/tree/main/tests) is blocked by the following issue on PyTorch 2.4 XPU backend:

* [pytorch/pytorch#127929]: xpu: support `torch.xpu.<memory>` ops (`memory_allocated`, `max_memory_allocated`, etc.)

Attempt to run Accelerate tests fails on tests collection.

[Intel® Data Center GPU Max 1100]: https://ark.intel.com/content/www/us/en/ark/products/232876/intel-data-center-gpu-max-1100.html

[pytorch/pytorch#127929]: https://github.com/pytorch/pytorch/issues/127929
[pytorch/pytorch#135550]: https://github.com/pytorch/pytorch/issues/135550

[pytorch/pytorch#135567]: https://github.com/pytorch/pytorch/pull/135567
[pytorch/pytorch#137886]: https://github.com/pytorch/pytorch/pull/137886

[intel/intel-xpu-backend-for-triton#2192]: https://github.com/intel/intel-xpu-backend-for-triton/pull/2192

[huggingface/accelerate#2825]: https://github.com/huggingface/accelerate/pull/2825

[huggingface/transformers#31237]: https://github.com/huggingface/transformers/issues/31237
