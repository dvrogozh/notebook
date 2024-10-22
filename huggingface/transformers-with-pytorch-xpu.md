# Running Transformers with PyTorch XPU backend

# Transformers

Haggingface Transformers support PyTorch XPU backend starting from version `4.42.0` (see [huggingface/transformers#31238]). Previously support for Intel Extension for PyTorch (IPEX) was available and got extended to cover direct integration of XPU backend in PyTorch (available starting from PyTorch 2.4).

# Testing

Transformers library can be formally tested for XPU backend by the project included tests on the system with supported Intel GPU graphics. To run tests first create `spec.py` file with the following contents:

```
import torch

DEVICE_NAME = 'xpu'

MANUAL_SEED_FN = torch.xpu.manual_seed
EMPTY_CACHE_FN = torch.xpu.empty_cache
DEVICE_COUNT_FN = torch.xpu.device_count
```

After that tests can be executed as follows:

```
cd /path/to/transformers/clone/copy
TRANSFORMERS_TEST_DEVICE_SPEC=spec.py python3 -m pytest --pspec tests/
```

# Using with PyTorch 2.5

XPU backend in PyTorch 2.5 greatly improved coverage of supported aten eager mode operators with just few missing for the executed Transformer tests. They are much easier to indentify now since [intel/torch-xpu-ops#318] is available in PyTorch 2.5 and adding `PYTORCH_ENABLE_XPU_FALLBACK=1` works fluently.

On Transformers `v4.45.2` and Accelerate `v0.34.2` select tests passing rates are as follows (executed on a system with a single [Intel® Data Center GPU Max 1100] card with `PYTORCH_ENABLE_XPU_FALLBACK=1`):

| Tests | Failed | Passed | Skipped |
| --- | --- | --- | --- |
| `tests/*.py` | 0 | 139 | 132 |
| `tests/benchmark` | 0 | 23 | 1 |
| `tests/generation` | 5 | 197 | 34 |
| `tests/models` | 5 | 32735 | 38838 |
| `tests/pipelines` | 0 | 158 | 141 |
| `tests/trainer` | 16 | 196 | 48 |
| `tests/utils` | 1 | 370 | 100 |

There are no more fatal Python errors on `tests/models/` tests, but there is a known hang issue on few `tests/models/levit/` tests:

* [pytorch/pytorch#136007]: xpu: huggingface levit tests hang on exit on PVC

### Missing ATen operators for XPU

Overall the following operators were noticed to fall back on PyTorch 2.5 XPU backend affecting Transformers library:

| Aten operation                      | CPU fallback |
| ----------------------------------- | ------------ |
| `aten::_ctc_loss`                   | Enabled      |
| `aten::_ctc_loss_backward`          | Enabled      |
| `aten::_fft_c2c`                    | Enabled      |
| `aten::_fft_c2r`                    | Enabled      |
| `aten::_fft_r2c`                    | Enabled      |
| `aten::index_copy.out`              | Enabled      |
| `aten::linspace.out`                | Enabled      |
| `aten::take`                        | Enabled      |
| `aten::upsample_bicubic2d_backward.grad_input` | Enabled |
| `aten::xlogy.OutTensor`             | Enabled      |

These are relevant requests to implement missing ATen operators:

* [pytorch/pytorch#127941]: xpu: set of unimplemented ops affect huggingface examples performance
* [pytorch/pytorch#136065]: xpu: set of aten ops are missing for Huggingface Transformers

### Missing PyTorch APIs

The following issues were **resolved** for PyTorch 2.5 providing essential APIs for Transformers and Accelerate libraries:

* [pytorch/pytorch#127929]: xpu: support `torch.xpu.<memory>` ops (`memory_allocated`, `max_memory_allocated`, etc.)
* [pytorch/pytorch#128478]: xpu: gradient checkpointing wrongly hits cuda path running on non-cuda devices

The following issue remains **open**. It prevents automated dispatching of the model loaded with `map_device="auto"` across multiple devices. This is an issue even with a single XPU capable card – model might be dispatched to CPU.

* [pytorch/pytorch#130599]: xpu: implement `torch.xpu.mem_get_info()` to support huggingface auto dispatch modes

Once issue will be resolved, it will affect behavior of those Transformer tests which rely on `map_device="auto"`. These tests might start to execute on XPU while previously ran on CPU and uncover more issues related to PyTorch XPU backend.

# Using with PyTorch 2.4

Huggingface Model Cards, Tranformer [examples](https://github.com/huggingface/transformers/tree/v4.45.2/examples/pytorch) and Transformer [tests](https://github.com/huggingface/transformers/tree/v4.45.2/tests) can be used to try out Transformers library with PyTorch 2.4 XPU backend. Note that tests are not entirely blocked by [pytorch/pytorch#127929] as it was with Accelerate.

It's important to note that a number of PyTorch eager mode operators are not yet implemented in PyTorch 2.4 for XPU backend. Such operators might (silently in PyTorch 2.4) fall back to CPU or cause script termination if operator is not marked for automated CPU fallback. In the latter case fallback can be forced by running with `PYTORCH_ENABLE_XPU_FALLBACK=1` environment variable. To make PyTorch print warnings on CPU fallback it's required to apply [intel/torch-xpu-ops#318] and run with `PYTORCH_DEBUG_XPU_FALLBACK=1` environment variable.

On Transformers `v4.45.2` and Accelerate `v0.34.2` select tests passing rates are as follows (executed on a system with a single [Intel® Data Center GPU Max 1100] card with `PYTORCH_ENABLE_XPU_FALLBACK=1`):

| Tests | Failed | Passed | Skipped |
| --- | --- | --- | --- |
| `tests/*.py` | 0 | 139 | 132 |
| `tests/benchmark` | 0 | 23 | 1 |
| `tests/generation` | 5 | 197 | 34 |
| `tests/models` | 113 | 32750 | 38540 |
| `tests/pipelines` | 0 | 158 | 141 |
| `tests/trainer` | 4 | 192 | 55 |
| `tests/utils` | 2 | 368 | 100 |

Tests in `tests/models/olmoe`, `tests/models/fastspeech2_conformer` and `tests/models/qwen2_moe` were excluded since fatal Python error is happening during the test causing test abortion.

See below key issues found trying out Huggingface Transformers with PyTorch XPU backend.

### Missing ATen operators for XPU

Overall the following operators were noticed to fall back on PyTorch 2.4 XPU backend affecting Transformers library:

* Running [Backbone Models]:

| Aten operation                      | CPU fallback |
| ----------------------------------- | ------------ |
| `aten::_adaptive_avg_pool2d`        | Enabled      |
| `aten::addcmul.out`                 | Enabled      |
| `aten::all.all_out`                 | Enabled      |
| `aten::ceil.out`                    | Enabled      |
| `aten::floor.out`                   | Enabled      |
| `aten::linalg_vector_norm.out`      | Enabled      |
| `aten::log_sigmoid_forward`         | Enabled      |
| `aten::masked_select`               | Enabled      |
| `aten::max_pool2d_with_indices.out` | Enabled      |
| `aten::mse_loss_backward`           | Enabled      |
| `aten::native_batch_norm`           | Enabled      |
| `aten::native_batch_norm_backward`  | Enabled      |
| `aten::native_group_norm_backward`  | Enabled      |
| `aten::nll_loss2d_forward`          | Manual       |
| `aten::nll_loss2d_backward`         | Manual       |
| `aten::norm.out`                    | Enabled      |
| `aten::roll`                        | Enabled      |
| `aten::sgn.out`                     | Enabled      |
| `aten::sigmoid.out`                 | Enabled      |
| `aten::sigmoid_backward.grad_input` | Enabled      |
| `aten::upsample_bilinear2d.out`     | Enabled      |
| `aten::upsample_bilinear2d_backward.grad_input` | Enabled |
| `aten::upsample_nearest2d.out`                  | Enabled |
| `aten::upsample_nearest2d_backward.grad_input`  | Enabled |

* Running [Examples]:

| Aten operation                       | CPU fallback |
| ------------------------------------ | ------------ |
| `aten::_cdist_forward`               | Enabled      |
| `aten::_foreach_addcdiv_.ScalarList` | Manual       |
| `aten::_foreach_addcmul_.Scalar`     | Manual       |
| `aten::_foreach_div_.ScalarList`     | Manual       |
| `aten::_foreach_lerp_.Scalar`        | Manual       |
| `aten::_foreach_mul_.Scalar`         | Manual       |
| `aten::_foreach_mul_.Tensor`         | Manual       |
| `aten::_foreach_norm.Scalar`         | Manual       |
| `aten::_foreach_sqrt`                | Manual       |
| `aten::addcdiv.out`                  | Enabled      |
| `aten::addcmul.out`                  | Enabled      |
| `aten::all.all_out`                  | Enabled      |
| `aten::floor.out`                    | Enabled      |
| `aten::grid_sampler_2d_backward`     | Enabled      |
| `aten::lerp.Scalar_out`              | Enabled      |
| `aten::linalg_vector_norm.out`       | Enabled      |
| `aten::linspace.out`                 | Enabled      |
| `aten::native_batch_norm`            | Enabled      |
| `aten::native_group_norm_backward`   | Enabled      |
| `aten::nll_loss2d_forward`           | Manual       |
| `aten::nll_loss2d_backward`          | Manual       |
| `aten::max_pool2d_with_indices.out`  | Enabled      |
| `aten::prod.int_out`                 | Enabled      |
| `aten::roll`                         | Enabled      |
| `aten::sgn.out`                      | Enabled      |
| `aten::sigmoid.out`                  | Enabled      |
| `aten::sigmoid_backward.grad_input`  | Enabled      |
| `aten::silu.out`                     | Enabled      |
| `aten::topk.values`                  | Enabled      |
| `aten::upsample_bilinear2d.out`      | Enabled      |
| `aten::upsample_bilinear2d_backward.grad_input` | Enabled |
| `aten::upsample_nearest2d.out`       | Enabled      |

* Running select tests for pipelines, trainers and models:

| Aten operation                 | CPU fallback |
| ------------------------------ | ------------ |
| `aten::addcdiv.out`            | Enabled      |
| `aten::addcmul.out`            | Enabled      |
| `aten::equal`                  | Enabled      |
| `aten::isin.Tensor_Tensor_out` | Enabled      |
| `aten::lerp.Scalar_out`        | Enabled      |
| `aten::linalg_vector_norm.out` | Enabled      |
| `aten::logical_or.out`         | Enabled      |
| `aten::mse_loss_backward`      | Enabled      |
| `aten::mse_loss.out`           | Enabled      |
| `aten::multinomial`            | Enabled      |
| `aten::norm.out`               | Enabled      |
| `aten::topk.values`            | Enabled      |

These are relevant requests to implement missing ATen operators:

* [pytorch/pytorch#127931]: xpu: a set of foreach ops not implemented for XPU backend affecting Huggingface examples
* [pytorch/pytorch#127937]: xpu: `aten::nll_loss2d_*` not implemented for XPU backend affecting Huggingface examples
* [pytorch/pytorch#127941]: xpu: set of unimplemented ops affect huggingface examples performance
* [pytorch/pytorch#128914]: xpu: set of not implemented aten ops affecting huggingface tests 

### Missing PyTorch APIs

The following missing PyTorch APIs affect running Transformers:

* [pytorch/pytorch#127929]: xpu: support `torch.xpu.<memory>` ops (`memory_allocated`, `max_memory_allocated`, etc.)
* [pytorch/pytorch#128478]: xpu: gradient checkpointing wrongly hits cuda path running on non-cuda devices
* [pytorch/pytorch#130599]: xpu: implement `torch.xpu.mem_get_info()` to support huggingface auto dispatch modes

Pay special attention on [pytorch/pytorch#130599]. It prevents automated dispatching of the model loaded with `map_device="auto"` across multiple devices. This is an issue even with a single XPU capable card – model might be dispatched to CPU.

[Backbone Models]: https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/backbones
[Examples]: https://github.com/huggingface/transformers/tree/v4.42.2/examples/pytorch

[pytorch/pytorch#127929]: https://github.com/pytorch/pytorch/issues/127929
[pytorch/pytorch#127931]: https://github.com/pytorch/pytorch/issues/127931
[pytorch/pytorch#127937]: https://github.com/pytorch/pytorch/issues/127937
[pytorch/pytorch#127941]: https://github.com/pytorch/pytorch/issues/127941
[pytorch/pytorch#128478]: https://github.com/pytorch/pytorch/issues/128478
[pytorch/pytorch#128914]: https://github.com/pytorch/pytorch/issues/128914
[pytorch/pytorch#130599]: https://github.com/pytorch/pytorch/issues/130599
[pytorch/pytorch#136007]: https://github.com/pytorch/pytorch/issues/136007
[pytorch/pytorch#136065]: https://github.com/pytorch/pytorch/issues/136065

[intel/torch-xpu-ops#318]: https://github.com/intel/torch-xpu-ops/pull/318

[huggingface/transformers#31238]: https://github.com/huggingface/transformers/pull/31238
