# Run LLama Stack on Intel GPU

## Overview

> [!WARNING]
> llama-stack is currently work in progress and is constantly evolving. Pay attention on the stack version (commit id or tag) being described below.

[llama-stack] provides building blocks to build llama applications. It contains API specifications, API providers and distributions. Distributions can be used to build llama stack servers to serve applications.

As of [91e7efb] llama-stack requires patches to support Intel GPUs via PyTorch XPU backend:

* Patches for [llama-stack] at [91e7efb]:

  * [0001-feat-enable-xpu-support-for-meta-reference-stack.patch]

* Patches for [llama-models] at [ec6b563]:

  * [0001-Add-optional-arg-to-specify-device-for-Transformer-m.patch]
  * [0002-Add-option-to-initialize-multimodal-model-on-devices.patch]

## Install `llama-stack`

llama-stack package provides `llama` cli tool to build and manage LLama Stack Distributions and models. Download, patch and install llama-stack as follows:

```
git clone https://github.com/meta-llama/llama-stack.git && cd llama-stack
git am $PATCHES/llama-stack/0001-feat-enable-xpu-support-for-meta-reference-stack.patch
pip install -e .
```

> [!NOTE]
> We could install with `pip install llama-stack` if we did not need to apply patches.

Once installed, `llama` cli will be available. We will use it to further setup and run LLama Server.

## Build LLama Stack Distributions

llama-stack provides a number of templates to build LLama Servers (Distributions). Custom configs are also possible. Each distribution can be built either in a form of Docker image or Conda environment. These are 2 prerequisites and at least one of them needs to be installed before LLama Stack Distributions can be built and used:

* Docker
* Conda

For the hints on how to install and use Conda, read [here](../conda/how-to-use-conda.md).

As of now Docker images are not available for PyTorch XPU backend. Below we will show how to build and setup Conda environments capable to work with XPU backend. 

## Build meta-reference stack

> [!NOTE]
> See [Meta Reference Distribution] for details.

Let's start with meta-reference-gpu stack which is actually designed to work with NVidia GPUs, but we will further adjust it to work with Intel GPUs. To build it, execute:

```
llama stack build --template meta-reference-gpu --image-type conda
```

Upon completion `llamastack-meta-reference-gpu` Conda virtual environment will be created. Enter environment and make the following customizations

* Patch and reinstall [llama-models] to enable non-CUDA devices as follows. These patches are taken from [llama-models#165] PR.

```
conda activate llamastack-meta-reference-gpu
git clone https://github.com/meta-llama/llama-models.git && cd llama-models
git reset --hard ec6b563
git am $PATCHES/llama-models/0001-Add-optional-arg-to-specify-device-for-Transformer-m.patch
git am $PATCHES/llama-models/0002-Add-option-to-initialize-multimodal-model-on-devices.patch
pip uninstall -y llama-models
pip install -e .
```

* Reinstall XPU capable version of PyTorch:

```
conda activate llamastack-meta-reference-gpu
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/xpu
```

After that meta-reference stack should be able to work on Intel GPUs supported by PyTorch XPU backend.

## Preparing to serve

> [!NOTE]
> Note that llama-stack model identifiers differ from those used at Huggingface side (for our model that's `Llama3.2-3B-Instruct` vs. `meta-llama/Llama-3.2-3B-Instruct`). Use `llama model list` to list models supported by llama-stack and their identifiers.
> When working with llama-stack API use llama-stack model identifiers.

Before starting the server download checkpoints for the model you plan to serve. Checkpoints can be downloaded from different sources. For example, from Huggingface:

```
llama download --model-id Llama3.2-3B-Instruct --source huggingface
```

## Serving

Once distribution is built and configured, start it as follows:


```
cd /path/to/llama-stack && llama stack run \
  ~/.llama/distributions/llamastack-meta-reference-gpu/meta-reference-gpu-run.yaml \
  --port 5001 \
  --env INFERENCE_MODEL=Llama3.2-3B-Instruct
```

Note that it's required for some reason to start the server from the llama-stack working copy. It did not work starting from arbitrary directory.

On successful run you will see the following output from the server:
```
+ /home/dvrogozh/miniforge3/envs/llamastack-meta-reference-gpu/bin/python -m llama_stack.distribution.server.server --yaml-config /home/dvrogozh/.llama/distributions/llamastack-meta-reference-gpu/meta-reference-gpu-run.yaml --port 5001 --env INFERENCE_MODEL=Llama3.2-3B-Instruct
Setting CLI environment variable INFERENCE_MODEL => Llama3.2-3B-Instruct
Resolved 12 providers
 inner-inference => meta-reference-inference
 inner-memory => faiss
 models => __routing_table__
 inference => __autorouted__
 inner-safety => llama-guard
 shields => __routing_table__
 safety => __autorouted__
 memory_banks => __routing_table__
 memory => __autorouted__
 agents => meta-reference
 telemetry => meta-reference
 inspect => __builtin__

Loading model `Llama3.2-3B-Instruct`
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Loaded in 4.09 seconds
Loaded model...
Models: Llama3.2-3B-Instruct served by meta-reference-inference

Serving API telemetry
 GET /alpha/telemetry/get-trace
 POST /alpha/telemetry/log-event
Serving API memory
 POST /alpha/memory/insert
 POST /alpha/memory/query
Serving API agents
 POST /alpha/agents/create
 POST /alpha/agents/session/create
 POST /alpha/agents/turn/create
 POST /alpha/agents/delete
 POST /alpha/agents/session/delete
 POST /alpha/agents/session/get
 POST /alpha/agents/step/get
 POST /alpha/agents/turn/get
Serving API inference
 POST /alpha/inference/chat-completion
 POST /alpha/inference/completion
 POST /alpha/inference/embeddings
Serving API memory_banks
 GET /alpha/memory-banks/get
 GET /alpha/memory-banks/list
 POST /alpha/memory-banks/register
 POST /alpha/memory-banks/unregister
Serving API shields
 GET /alpha/shields/get
 GET /alpha/shields/list
 POST /alpha/shields/register
Serving API models
 GET /alpha/models/get
 GET /alpha/models/list
 POST /alpha/models/register
 POST /alpha/models/unregister
Serving API inspect
 GET /alpha/health
 GET /alpha/providers/list
 GET /alpha/routes/list
Serving API safety
 POST /alpha/safety/run-shield

Listening on ['::', '0.0.0.0']:5001
INFO:     Started server process [1501449]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:5001 (Press CTRL+C to quit)
```

To verify that server really handles incoming requests, run the following:
```
curl http://localhost:5001/alpha/inference/chat-completion \
-H "Content-Type: application/json" \
-d '{
    "model_id": "Llama3.2-3B-Instruct",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write me a 2 sentence poem about the moon"}
    ],
    "sampling_params": {"temperature": 0.7, "seed": 42, "max_tokens": 512}
}'
```

The output will be similar to the following (will be on a single line vs. what is shown below):

```
{
  "completion_message": {
    "role": "assistant",
    "content": "Here is a 2 sentence poem about the moon:\n\nSilent crescent in the midnight sky,\nA glowing beacon, passing us by.",
    "stop_reason": "end_of_turn",
    "tool_calls": []
    },
  "logprobs": null
}
```

[llama-stack]: https://github.com/meta-llama/llama-stack
[91e7efb]: https://github.com/meta-llama/llama-stack/commit/91e7efbc91c729d74c5cf9b3947d3e8acc1fbb71
[0001-feat-enable-xpu-support-for-meta-reference-stack.patch]: patches/llama-stack/0001-feat-enable-xpu-support-for-meta-reference-stack.patch

[llama-models]: https://github.com/meta-llama/llama-models
[ec6b563]: https://github.com/meta-llama/llama-models/commit/ec6b56330258f6c544a6ca95c52a2aee09d8e3ca

[llama-models#165]: https://github.com/meta-llama/llama-models/pull/165

[0001-Add-optional-arg-to-specify-device-for-Transformer-m.patch]: patches/llama-models/0001-Add-optional-arg-to-specify-device-for-Transformer-m.patch
[0002-Add-option-to-initialize-multimodal-model-on-devices.patch]: patches/llama-models/0002-Add-option-to-initialize-multimodal-model-on-devices.patch

[Meta Reference Distribution]: https://github.com/meta-llama/llama-stack/blob/91e7efbc91c729d74c5cf9b3947d3e8acc1fbb71/docs/source/getting_started/distributions/self_hosted_distro/meta-reference-gpu.md
