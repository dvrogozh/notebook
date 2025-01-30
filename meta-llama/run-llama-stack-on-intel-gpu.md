# Run LLama Stack on Intel GPU

## Overview

> [!WARNING]
> llama-stack is currently work in progress and is constantly evolving. Pay attention on the stack version (commit id or tag) being described below.

[llama-stack] provides building blocks to build llama applications. It contains API specifications, API providers and distributions. Distributions can be used to build llama stack servers to serve applications.

As of [7fe25927] llama-stack requires patches to support Intel GPUs via PyTorch XPU backend:

* [0001-feat-enable-xpu-support-for-meta-reference-stack.patch] (from [llama-stack#558])

## Install `llama-stack`

llama-stack package provides `llama` cli tool to build and manage LLama Stack Distributions and models. Download, patch and install llama-stack as follows (patch taken from [llama-stack#558] PR):

```
git clone https://github.com/meta-llama/llama-stack.git && cd llama-stack
git am $PATCHES/llama-stack/0001-feat-enable-xpu-support-for-meta-reference-stack.patch
pip install -e .
```

> [!NOTE]
> We could install with `pip install llama-stack` if we did not need to apply patches.

Once installed, `llama` cli will be available. We will use it to further setup and run LLama Server.

## Preparing to serve

> [!NOTE]
> Note that llama-stack model identifiers differ from those used at Huggingface side (for our model that's `Llama3.2-3B-Instruct` vs. `meta-llama/Llama-3.2-3B-Instruct`). Use `llama model list` to list models supported by llama-stack and their identifiers.
> When working with llama-stack API use llama-stack model identifiers.

Before starting the server download checkpoints for the model you plan to serve. Checkpoints can be downloaded from different sources. For example, from Huggingface:

```
llama download --model-id Llama3.2-3B-Instruct --source huggingface
```

## Build and run LLama Stack Distributions

llama-stack provides a number of templates to build LLama Servers (Distributions). Custom configs are also possible. Each distribution can be built either in a form of Docker image or Conda environment. These are 2 prerequisites and at least one of them needs to be installed before LLama Stack Distributions can be built and used:

* Docker
* Conda

For the hints on how to install and use Conda, read [here](../conda/how-to-use-conda.md).

As of now Docker images are not available for PyTorch XPU backend. Below we will show how to build and setup Conda environments capable to work with XPU backend. 

### Build meta-reference stack

> [!NOTE]
> See [Meta Reference Distribution] for details.

Let's start with meta-reference-gpu stack which is actually designed to work with NVidia GPUs, but we will further adjust it to work with Intel GPUs. To build it, execute:

```
llama stack build --template meta-reference-gpu --image-type conda --image-name llama-stack-xpu
```

Upon completion `llama-stack-xpu` Conda virtual environment will be created. Enter environment and make the following customizations:

* Reinstall [llama-models] with [224d48c] or later to enable non-CUDA devices:

```
conda activate llama-stack-xpu
git clone https://github.com/meta-llama/llama-models.git && cd llama-models
git reset --hard 224d48c
pip uninstall -y llama-models
pip install -e .
```

* Reinstall XPU capable version of PyTorch:

```
conda activate llama-stack-xpu
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/xpu
```

After that meta-reference stack should be able to work on Intel GPUs supported by PyTorch XPU backend. Start it as follows:

```
cd /path/to/llama-stack && llama stack run \
  ~/.llama/distributions/meta-reference-gpu/meta-reference-gpu-run.yaml \
  --image-name llama-stack-xpu \
  --port 5001 \
  --env INFERENCE_MODEL=Llama3.2-3B-Instruct
```

Note that it's required for some reason to start the server from the llama-stack working copy. It doesn't work starting from arbitrary directory.

On successful run you will see output from the server similar to the following:
```
Using run configuration: /home/dvrogozh/.llama/distributions/meta-reference-gpu/meta-reference-gpu-run.yaml
Using conda environment: llama-stack
+ /home/dvrogozh/miniforge3/envs/llama-stack/bin/python -m llama_stack.distribution.server.server --yaml-config /home/dvrogozh/.llama/distributions/meta-reference-gpu/meta-reference-gpu-run.yaml --port 5001 --env INFERENCE_MODEL=Llama3.2-3B-Instruct
Setting CLI environment variable INFERENCE_MODEL => Llama3.2-3B-Instruct
Using config file: /home/dvrogozh/.llama/distributions/meta-reference-gpu/meta-reference-gpu-run.yaml
Run configuration:
apis:
- agents
- datasetio
- eval
- inference
- safety
- scoring
- telemetry
- tool_runtime
- vector_io
container_image: null
datasets: []
eval_tasks: []
image_name: meta-reference-gpu
metadata_store:
  db_path: /home/dvrogozh/.llama/distributions/meta-reference-gpu/registry.db
  namespace: null
  type: sqlite
models:
- metadata: {}
  model_id: Llama3.2-3B-Instruct
  model_type: !!python/object/apply:llama_stack.apis.models.models.ModelType
  - llm
  provider_id: meta-reference-inference
  provider_model_id: null
- metadata:
    embedding_dimension: 384
  model_id: all-MiniLM-L6-v2
  model_type: !!python/object/apply:llama_stack.apis.models.models.ModelType
  - embedding
  provider_id: sentence-transformers
  provider_model_id: null
providers:
  agents:
  - config:
      persistence_store:
        db_path: /home/dvrogozh/.llama/distributions/meta-reference-gpu/agents_store.db
        namespace: null
        type: sqlite
    provider_id: meta-reference
    provider_type: inline::meta-reference
  datasetio:
  - config: {}
    provider_id: huggingface
    provider_type: remote::huggingface
  - config: {}
    provider_id: localfs
    provider_type: inline::localfs
  eval:
  - config: {}
    provider_id: meta-reference
    provider_type: inline::meta-reference
  inference:
  - config:
      checkpoint_dir: 'null'
      max_seq_len: 4096
      model: Llama3.2-3B-Instruct
    provider_id: meta-reference-inference
    provider_type: inline::meta-reference
  - config: {}
    provider_id: sentence-transformers
    provider_type: inline::sentence-transformers
  safety:
  - config: {}
    provider_id: llama-guard
    provider_type: inline::llama-guard
  scoring:
  - config: {}
    provider_id: basic
    provider_type: inline::basic
  - config: {}
    provider_id: llm-as-judge
    provider_type: inline::llm-as-judge
  - config:
      openai_api_key: '********'
    provider_id: braintrust
    provider_type: inline::braintrust
  telemetry:
  - config:
      service_name: llama-stack
      sinks: console,sqlite
      sqlite_db_path: /home/dvrogozh/.llama/distributions/meta-reference-gpu/trace_store.db
    provider_id: meta-reference
    provider_type: inline::meta-reference
  tool_runtime:
  - config:
      api_key: '********'
      max_results: 3
    provider_id: brave-search
    provider_type: remote::brave-search
  - config:
      api_key: '********'
      max_results: 3
    provider_id: tavily-search
    provider_type: remote::tavily-search
  - config: {}
    provider_id: code-interpreter
    provider_type: inline::code-interpreter
  - config: {}
    provider_id: rag-runtime
    provider_type: inline::rag-runtime
  - config: {}
    provider_id: model-context-protocol
    provider_type: remote::model-context-protocol
  vector_io:
  - config:
      kvstore:
        db_path: /home/dvrogozh/.llama/distributions/meta-reference-gpu/faiss_store.db
        namespace: null
        type: sqlite
    provider_id: faiss
    provider_type: inline::faiss
scoring_fns: []
shields: []
tool_groups:
- args: null
  mcp_endpoint: null
  provider_id: tavily-search
  toolgroup_id: builtin::websearch
- args: null
  mcp_endpoint: null
  provider_id: rag-runtime
  toolgroup_id: builtin::rag
- args: null
  mcp_endpoint: null
  provider_id: code-interpreter
  toolgroup_id: builtin::code_interpreter
vector_dbs: []
version: '2'

Warning: `bwrap` is not available. Code interpreter tool will not work correctly.
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Serving API scoring
 POST /v1/scoring/score
 POST /v1/scoring/score-batch
Serving API models
 GET /v1/models/{model_id}
 GET /v1/models
 POST /v1/models
 DELETE /v1/models/{model_id}
Serving API eval_tasks
 GET /v1/eval-tasks/{eval_task_id}
 GET /v1/eval-tasks
 POST /v1/eval-tasks
Serving API safety
 POST /v1/safety/run-shield
Serving API eval
 POST /v1/eval/tasks/{task_id}/evaluations
 DELETE /v1/eval/tasks/{task_id}/jobs/{job_id}
 GET /v1/eval/tasks/{task_id}/jobs/{job_id}/result
 GET /v1/eval/tasks/{task_id}/jobs/{job_id}
 POST /v1/eval/tasks/{task_id}/jobs
Serving API shields
 GET /v1/shields/{identifier}
 GET /v1/shields
 POST /v1/shields
Serving API tool_groups
 GET /v1/tools/{tool_name}
 GET /v1/toolgroups/{toolgroup_id}
 GET /v1/toolgroups
 GET /v1/tools
 POST /v1/toolgroups
 DELETE /v1/toolgroups/{toolgroup_id}
Serving API inspect
 GET /v1/health
 GET /v1/inspect/providers
 GET /v1/inspect/routes
 GET /v1/version
Serving API telemetry
 GET /v1/telemetry/traces/{trace_id}/spans/{span_id}
 GET /v1/telemetry/spans/{span_id}/tree
 GET /v1/telemetry/traces/{trace_id}
 POST /v1/telemetry/events
 GET /v1/telemetry/spans
 GET /v1/telemetry/traces
 POST /v1/telemetry/spans/export
Serving API vector_io
 POST /v1/vector-io/insert
 POST /v1/vector-io/query
Serving API vector_dbs
 GET /v1/vector-dbs/{vector_db_id}
 GET /v1/vector-dbs
 POST /v1/vector-dbs
 DELETE /v1/vector-dbs/{vector_db_id}
Serving API datasetio
 POST /v1/datasetio/rows
 GET /v1/datasetio/rows
Serving API agents
 POST /v1/agents
 POST /v1/agents/{agent_id}/session
 POST /v1/agents/{agent_id}/session/{session_id}/turn
 DELETE /v1/agents/{agent_id}
 DELETE /v1/agents/{agent_id}/session/{session_id}
 GET /v1/agents/{agent_id}/session/{session_id}
 GET /v1/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}
 GET /v1/agents/{agent_id}/session/{session_id}/turn/{turn_id}
Serving API tool_runtime
 POST /v1/tool-runtime/invoke
 GET /v1/tool-runtime/list-tools
 POST /v1/tool-runtime/rag-tool/insert
 POST /v1/tool-runtime/rag-tool/query
Serving API scoring_functions
 GET /v1/scoring-functions/{scoring_fn_id}
 GET /v1/scoring-functions
 POST /v1/scoring-functions
Serving API datasets
 GET /v1/datasets/{dataset_id}
 GET /v1/datasets
 POST /v1/datasets
 DELETE /v1/datasets/{dataset_id}
Serving API inference
 POST /v1/inference/chat-completion
 POST /v1/inference/completion
 POST /v1/inference/embeddings

Listening on ['::', '0.0.0.0']:5001
INFO:     Started server process [3561153]
INFO:     Waiting for application startup.
INFO:     ASGI 'lifespan' protocol appears unsupported.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:5001 (Press CTRL+C to quit)
```

### Build TGI stack

> [!NOTE]
> See [TGI Distribution] for details.

TGI Distribution assumes that TGI server was previously configured and started. Follow [Running TGI with PyTorch XPU backend](../huggingface/tgi-with-pytorch-xpu.md) to start the one for Intel GPU. We will assume that TGI was started for Llama3.2-3B-Instruct model:

```
text-generation-launcher --model-id meta-llama/Llama-3.2-3B-Instruct --port 8080
```

Next, build TGI llama-stack Distribution:

```
llama stack build --template tgi --image-type conda --image-name llama-stack-tgi
```

Since inference will be executed by TGI server it's not needed to further configure TGI llama-stack conda environment for XPU. At this point it's ready to serve. Start serving with:

```
cd /path/to/llama-stack && llama stack run \
  ~/.llama/distributions/llamastack-tgi/tgi-run.yaml \
  --image-name llama-stack-tgi \
  --port 5001 \
  --env INFERENCE_MODEL=Llama3.2-3B-Instruct \
  --env TGI_URL=http://127.0.0.1:8080
```

On successful run end of the output should be similar to the following:

```
...
Listening on ['::', '0.0.0.0']:5001
INFO:     Started server process [1005804]
INFO:     Waiting for application startup.
INFO:     ASGI 'lifespan' protocol appears unsupported.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://['::', '0.0.0.0']:5001 (Press CTRL+C to quit)
```

## Verifying the server

`llama-stack-client` can be used to connect to llama server. It gets installed as a dependency
of `llama-stack` package we've initially installed. To use the client first configure the llama stack endpoint:

```
llama-stack-client configure --endpoint http://localhost:5001
```

To verify that server really handles incoming requests, run the following:

* To query list of available models:

```
# llama-stack-client models list
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ identifier         ┃ provider_id        ┃ provider_resource… ┃ metadata            ┃ model_type ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Llama3.2-3B-Instr… │ meta-reference-in… │ Llama3.2-3B-Instr… │ {}                  │ llm        │
│ all-MiniLM-L6-v2   │ sentence-transfor… │ all-MiniLM-L6-v2   │ {'embedding_dimens… │ embedding  │
│                    │                    │                    │ 384.0}              │            │
└────────────────────┴────────────────────┴────────────────────┴─────────────────────┴────────────┘
```

* To run inference with the llama stack:

```
# llama-stack-client   inference chat-completion   --message "hello, what model are you?"
ChatCompletionResponse(
    completion_message=CompletionMessage(
        content='Hello! I\'m an AI model, specifically a type of conversational AI designed to
understand and respond to human language. I\'m a large language model, which means I was trained on
a massive dataset of text from various sources, including books, articles, and online
conversations.\n\nI don\'t have a specific "model" in the classical sense, but I\'m based on a type
of neural network architecture called a transformer. This architecture is particularly well-suited
for natural language processing tasks, such as understanding and generating human language.\n\nI\'m
a cloud-based model, which means I\'m hosted on a network of servers and can be accessed through
the internet. I don\'t have a physical body or consciousness, but I\'m designed to simulate
conversation and answer questions to the best of my ability based on my training data.\n\nHow can I
help you today?',
        role='assistant',
        stop_reason='end_of_turn',
        tool_calls=[]
    ),
    logprobs=None
)
```

[llama-stack]: https://github.com/meta-llama/llama-stack
[7fe25927]: https://github.com/meta-llama/llama-stack/commit/7fe25927954d0ac00901091e3a01d06fc0ef09c9
[0001-feat-enable-xpu-support-for-meta-reference-stack.patch]: patches/llama-stack/0001-feat-enable-xpu-support-for-meta-reference-stack.patch
[llama-stack#558]: https://github.com/meta-llama/llama-stack/pull/558

[llama-models]: https://github.com/meta-llama/llama-models
[224d48c]: https://github.com/meta-llama/llama-models/commit/224d48ca38c985dc77e79a842d5e1e7a5c6832f3
[llama-models#233]: https://github.com/meta-llama/llama-models/pull/233

[Meta Reference Distribution]: https://github.com/meta-llama/llama-stack/blob/7fe25927954d0ac00901091e3a01d06fc0ef09c9/docs/source/distributions/self_hosted_distro/meta-reference-gpu.md
[TGI Distribution]: https://github.com/meta-llama/llama-stack/blob/7fe25927954d0ac00901091e3a01d06fc0ef09c9/docs/source/distributions/self_hosted_distro/tgi.md
