# Run LLama Stack on Intel GPU

## Overview

[llama-stack] provides building blocks to build llama applications. It contains API specifications, API providers and distributions. Distributions can be used to build llama stack servers to serve applications.

As of [0784284] llama-stack requires patches to support Intel GPUs via PyTorch XPU backend:

* Patches for [llama-stack] at [0784284]:

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
pip uninstall llama-models
pip install -e .
```

* Reinstall XPU capable version of PyTorch:

```
conda activate llamastack-meta-reference-gpu
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/xpu
```

After that meta-reference stack should be able to work on Intel GPUs supported by PyTorch XPU backend.

## Preparing to serve

Before starting the server do the following:

* Download checkpoints for the model you plan to serve. Checkpoints can be downloaded from different sources. For example, using Huggingface as a source:

```
llama download --model-id Llama3.2-3B-Instruct --source huggingface
```

* Adjust distribution configuration per your needs to be able to serve. This might include the following:

  * Consider to comment out `chromadb`, `pgvector` and/or other memory providers which you don't plan to use. Not doing so will result in a failure to start LLama Server if the service associated with memory provider won't be found. It's enough to have just `inline::faiss` memory provider if you plan experiments local to your system.

  * Add model to the list of served models following [this description](https://github.com/meta-llama/llama-stack-apps/pull/114#pullrequestreview-2436905656). Otherwise server won't be able to handle incoming requests.

Overall, here is a distributions configuration which worked:

```
$ cat ~/.llama/distributions/llamastack-meta-reference-gpu/meta-reference-gpu-run.yaml
version: '2'
built_at: '2024-11-19T14:04:22.700025'
image_name: meta-reference-gpu
docker_image: null
conda_env: meta-reference-gpu
apis:
- inference
- memory
- safety
- agents
- telemetry
providers:
  inference:
  - provider_id: inline::meta-reference-0
    provider_type: inline::meta-reference
    config:
      #model: Llama3.2-3B-Instruct
      torch_seed: null
      max_seq_len: 4096
      max_batch_size: 1
      create_distributed_process_group: true
      checkpoint_dir: null
  memory:
  - provider_id: inline::faiss-0
    provider_type: inline::faiss
    config:
      kvstore:
        namespace: null
        type: sqlite
        db_path: /home/dvrogozh/.llama/runtime/faiss_store.db
  #- provider_id: remote::chromadb-1
  #  provider_type: remote::chromadb
  #  config:
  #    host: localhost
  #    port: null
  #    protocol: http
  #- provider_id: remote::pgvector-2
  #  provider_type: remote::pgvector
  #  config:
  #    host: localhost
  #    port: 5432
  #    db: postgres
  #    user: postgres
  #    password: mysecretpassword
  safety:
  - provider_id: inline::llama-guard-0
    provider_type: inline::llama-guard
    config:
      excluded_categories: []
  agents:
  - provider_id: inline::meta-reference-0
    provider_type: inline::meta-reference
    config:
      persistence_store:
        namespace: null
        type: sqlite
        db_path: /home/dvrogozh/.llama/runtime/kvstore.db
  telemetry:
  - provider_id: inline::meta-reference-0
    provider_type: inline::meta-reference
    config: {}
metadata_store: null
models:
  - model_id: Llama3.2-3B-Instruct
    provider_id: inline::meta-reference-0
shields: []
memory_banks: []
datasets: []
scoring_fns: []
eval_tasks: []
```

## Serving

Once distribution is built and configured, start it as follows:


```
cd /path/to/llama-stack && llama stack run \
  ~/.llama/distributions/llamastack-meta-reference-gpu/meta-reference-gpu-run.yaml
```

Note that it's required for some reason to start the server from the llama-stack working copy. It did not work starting from arbitrary directory.

On successful run you will see the following output from the server:
```
Resolved 12 providers
 inner-inference => inline::meta-reference-0
 inner-memory => inline::faiss-0
 models => __routing_table__
 inference => __autorouted__
 inner-safety => inline::llama-guard-0
 shields => __routing_table__
 safety => __autorouted__
 memory_banks => __routing_table__
 memory => __autorouted__
 agents => inline::meta-reference-0
 telemetry => inline::meta-reference-0
 inspect => __builtin__

Loading model `Llama3.2-3B-Instruct`
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Loaded in 4.33 seconds
Loaded model...
`Llama3.2-3B-Instruct` already registered with `inline::meta-reference-0`
Models: Llama3.2-3B-Instruct served by inline::meta-reference-0

Serving API shields
 GET /shields/get
 GET /shields/list
 POST /shields/register
Serving API safety
 POST /safety/run_shield
Serving API agents
 POST /agents/create
 POST /agents/session/create
 POST /agents/turn/create
 POST /agents/delete
 POST /agents/session/delete
 POST /agents/session/get
 POST /agents/step/get
 POST /agents/turn/get
Serving API memory_banks
 GET /memory_banks/get
 GET /memory_banks/list
 POST /memory_banks/register
 POST /memory_banks/unregister
Serving API models
 GET /models/get
 GET /models/list
 POST /models/register
 POST /models/unregister
Serving API inspect
 GET /health
 GET /providers/list
 GET /routes/list
Serving API memory
 POST /memory/insert
 POST /memory/query
Serving API inference
 POST /inference/chat_completion
 POST /inference/completion
 POST /inference/embeddings
Serving API telemetry
 GET /telemetry/get_trace
 POST /telemetry/log_event

Listening on ['::', '0.0.0.0']:5000
INFO:     Started server process [1397529]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:5000 (Press CTRL+C to quit)
```

To verify that server really handles incoming requests, run the following:
```
curl http://localhost:5000/inference/chat_completion \
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
[0784284]: https://github.com/meta-llama/llama-stack/commit/0784284ab582ec864a0a203102c2aaac110d54be
[0001-feat-enable-xpu-support-for-meta-reference-stack.patch]: patches/llama-stack/0001-feat-enable-xpu-support-for-meta-reference-stack.patch

[llama-models]: https://github.com/meta-llama/llama-models
[ec6b563]: https://github.com/meta-llama/llama-models/commit/ec6b56330258f6c544a6ca95c52a2aee09d8e3ca

[llama-models#165]: https://github.com/meta-llama/llama-models/pull/165

[0001-Add-optional-arg-to-specify-device-for-Transformer-m.patch]: patches/llama-models/0001-Add-optional-arg-to-specify-device-for-Transformer-m.patch
[0002-Add-option-to-initialize-multimodal-model-on-devices.patch]: patches/llama-models/0002-Add-option-to-initialize-multimodal-model-on-devices.patch
