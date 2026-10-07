# Live integration acceptance

## PostgreSQL hybrid-index manifest

The optional contract suite checks shared lease ownership, fencing, activation, and retirement
against a live server. Install the extra, set a test database DSN, and run:

```sh
uv sync --all-groups --extra postgres
GABBY_POSTGRES_DSN='postgresql://user:password@localhost/test_db' \
  uv run --frozen pytest tests/integration/test_postgres_indexing_live.py -q
```

The test creates and drops a uniquely named schema in the configured database and applies
`sql/postgres_generation_manifest.sql`. Use a disposable test database and a role allowed to create
schemas. The adapter assumes the host has already applied the migration in production.

## PostgreSQL shared knowledge store

The opt-in store suite checks full-text retrieval, exact metadata filters, transactional source
replacement, generation filtering, and stale-writer fencing against PostgreSQL. Run it with the
same disposable database and `postgres` extra:

```sh
uv sync --all-groups --extra postgres
GABBY_POSTGRES_DSN='postgresql://user:password@localhost/test_db' \
  uv run --frozen pytest tests/integration/test_postgres_knowledge_live.py -q
```

The suite creates and removes an isolated schema and applies `sql/postgres_knowledge.sql`.

## PostgreSQL shared vector store

The opt-in vector suite requires PostgreSQL with pgvector 0.8.0 or newer installed and a
database role allowed to create the extension when needed, create schemas, and drop them. It checks
cosine ordering, exact metadata filters, source replacement, generation visibility, dimension
consistency, stale-writer rejection, and cleanup:

```sh
uv sync --all-groups --extra postgres
GABBY_POSTGRES_DSN='postgresql://user:password@localhost/test_db' \
  uv run --frozen pytest tests/integration/test_postgres_vector_live.py -q
```

The test creates and removes an isolated schema and applies `sql/postgres_vector.sql`. Use a
disposable test database; the migration installs pgvector in `public` if it is not already present.
The reusable PostgreSQL acceptance workflow runs this suite against the versioned
`pgvector/pgvector:0.8.6-pg17-trixie` service image.

## PostgreSQL resumable SSE journal

The optional journal suite checks shared session capacity, principal/request binding, bounded
ordered replay, completion, and abandoned-run recovery across pool connections. Use the same
`GABBY_POSTGRES_DSN` and optional extra, then run:

```sh
uv sync --all-groups --extra postgres
GABBY_POSTGRES_DSN='postgresql://user:password@localhost/test_db' \
  uv run --frozen pytest tests/integration/test_postgres_stream_journal_live.py -q
```

It creates an isolated schema and applies `sql/postgres_stream_journal.sql`. Use a disposable test
database and a role allowed to create schemas. Production hosts apply the migration themselves.

## Container sandbox

These checks use a real Docker or Podman Linux runtime and are separate from the ordinary test
suite. They verify effective container settings, disabled network access, workspace mount modes,
shell and custom-tool execution, sandboxed Python source execution through the private tool-input
mount, cleanup after normal completion, and cleanup when an active run is cancelled.
The tests require the matching CLI and daemon; API-adapter checks also exercise the container API.

Run the complete Linux sandbox matrix with:

```sh
GABBY_RUN_DOCKER_INTEGRATION=1 GABBY_RUN_PODMAN_INTEGRATION=1 \
uv run --frozen pytest tests/integration/test_docker_sandbox_live.py \
  tests/integration/test_podman_sandbox_live.py -q
```

The matrix uses `python:3.14-slim`, pulling it when needed. Each run creates temporary containers
and workspaces, and asserts that each created container has been removed. Host/engine combinations
other than the Linux Docker and Podman runs still need separate acceptance evidence.
Native Windows Docker runs additionally require Docker Engine 29.1.4 or newer; Gabby checks this
before starting a Windows container with networking disabled.

The Docker API checks discover a local Unix socket from `DOCKER_HOST=unix://...`, then try
`/var/run/docker.sock` and Docker Desktop's `~/.docker/run/docker.sock`. Set
`GABBY_DOCKER_API_SOCKET` to override discovery when a host uses another local socket path. On
macOS with Docker Desktop, run the same Docker acceptance with:

```sh
GABBY_DOCKER_API_SOCKET="$HOME/.docker/run/docker.sock" \
GABBY_RUN_DOCKER_INTEGRATION=1 \
uv run --frozen pytest tests/integration/test_docker_sandbox_live.py -q
```

This command exercises both the Docker CLI and API adapters on the macOS host. It is acceptance
coverage to run on macOS; a passing Linux Docker run does not verify it.

## Native Windows Docker

Run this acceptance on a Windows host with the Hyper-V feature enabled, a Windows-container Docker
daemon at version 29.1.4 or newer, and an image compatible with that host. The image must contain
PowerShell and `cmd.exe`; read/write workspace acceptance also requires the image user to have
permission to write to the temporary bind mount. Gabby checks Hyper-V isolation, CPU and memory
settings, network denial from inside the container, read-only and read/write workspace behavior,
and per-run cleanup, including cancellation. These checks are opt-in. The API adapter connects to
Docker Desktop's named pipe and requires the `windows-sandbox` extra:

```powershell
uv sync --extra windows-sandbox
$env:GABBY_RUN_WINDOWS_DOCKER_INTEGRATION = "1"
$env:GABBY_WINDOWS_DOCKER_IMAGE = "mcr.microsoft.com/windows/servercore:ltsc2022"
$env:GABBY_WINDOWS_DOCKER_NAMED_PIPE = "\\.\pipe\docker_engine"
uv run --frozen pytest tests/integration/test_windows_docker_sandbox_live.py -q
```

Choose a Windows image tag supported by the host and ensure Docker can pull it. The test does not
configure the Docker daemon or install/enable Hyper-V for you. It runs the isolation, mount, network,
and cleanup checks through both the CLI adapter and the API adapter over the named pipe. The API
adapter requires the standard Proactor event loop used by Windows Python; alternate event loops that
do not support named pipes are rejected by the transport.

## Live model-provider acceptance

These checks contact real model providers, so they are opt-in and skipped in the normal test
suite. Hosted providers receive one completion and one streaming completion; local Transformers
acceptance loads a selected checkpoint and runs one bounded completion. Use a model endpoint,
checkpoint, and account approved for the prompts used by these smoke tests.

## Cohere reranking

The reranker acceptance is opt-in and verifies the live API contract and candidate identity mapping.
It does not measure ranking quality. Configure `COHERE_API_KEY` and optionally select a model:

```sh
GABBY_RUN_COHERE_RERANK_INTEGRATION=1 \
GABBY_COHERE_RERANK_MODEL=rerank-v4.0-fast \
uv run --frozen pytest tests/integration/test_cohere_reranker_live.py -q
```

The adapter defaults to `https://api.cohere.com/v2/rerank`, sends candidate text to Cohere, and does
not send source metadata or document IDs. The model and endpoint must be enabled for the account.

## Jina reranking

This opt-in check verifies a live ranking request and candidate identity mapping; it does not measure
general ranking quality. Set `JINA_API_KEY` and optionally select a model:

```sh
GABBY_RUN_JINA_RERANK_INTEGRATION=1 \
GABBY_JINA_RERANK_MODEL=jina-reranker-v3.5 \
uv run --frozen pytest tests/integration/test_jina_reranker_live.py -q
```

The adapter posts candidate text to Jina's `https://api.jina.ai/v1/rerank` endpoint. Review [Jina's
reranker API](https://jina.ai/en-US/reranker/) and account terms before enabling the check; usage
charges may apply.

## Voyage reranking

This opt-in check verifies a live ranking request and candidate identity mapping; it does not measure
general ranking quality. Set `VOYAGE_API_KEY` and run:

```sh
GABBY_RUN_VOYAGE_RERANK_INTEGRATION=1 \
GABBY_VOYAGE_RERANK_MODEL=rerank-2.5-lite \
uv run --frozen pytest tests/integration/test_voyage_reranker_live.py -q
```

The adapter posts to `https://api.voyageai.com/v1/rerank`, asks for index-only results, and disables
provider-side truncation by default. Candidate text is sent to Voyage; the live check may incur
account usage charges. Review Voyage's [reranker API reference](https://docs.voyageai.com/reference/reranker-api)
and account terms before enabling it.

## NVIDIA reranking

This opt-in contract check sends candidate text to NVIDIA and verifies ranking and candidate identity
mapping; it does not measure general ranking quality. Set `NVIDIA_API_KEY` and run:

```sh
GABBY_RUN_NVIDIA_RERANK_INTEGRATION=1 \
GABBY_NVIDIA_RERANK_MODEL=nvidia/rerank-qa-mistral-4b \
uv run --frozen pytest tests/integration/test_nvidia_reranker_live.py -q
```

The default endpoint is NVIDIA's hosted NeMo reranking API. Review [its API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nim-rerankqa-mistral-4b-v3-infer)
and service terms before enabling the check; account usage charges may apply.

## Local Transformers reranking

This opt-in check loads a sequence-classification checkpoint and verifies bounded ranking output
and candidate identity mapping. It does not measure relevance quality. Install Gabby's `transformers`
extra and a host-compatible PyTorch build first; choose a reviewed checkpoint and pin its Hub commit.

```sh
GABBY_RUN_TRANSFORMERS_RERANKER_INTEGRATION=1 \
GABBY_TRANSFORMERS_RERANKER_MODEL=your-org/your-cross-encoder \
GABBY_TRANSFORMERS_RERANKER_REVISION=your-reviewed-commit \
GABBY_TRANSFORMERS_RERANKER_LOCAL_FILES_ONLY=1 \
GABBY_TRANSFORMERS_DEVICE=cpu \
uv run --frozen pytest tests/integration/test_transformers_reranker_live.py -q
```

The model must be a local or cached Hugging Face sequence classifier. Set
`GABBY_TRANSFORMERS_RERANKER_LOCAL_FILES_ONLY=0` for the first run when downloading is intended;
subsequent checks can use `1` to stay offline. Private checkpoints can use `HF_TOKEN` through the
host environment. Gabby disables custom model code and requires safetensors weights.

## Ollama

Make sure Ollama is running and the selected model is available locally, then run:

```sh
GABBY_RUN_OLLAMA_INTEGRATION=1 \
GABBY_OLLAMA_MODEL=your-local-model \
uv run --frozen pytest tests/integration/test_model_providers_live.py -k ollama -q
```

The adapter defaults to `http://localhost:11434/v1`. Set `GABBY_OLLAMA_BASE_URL` to use another
endpoint and `GABBY_OLLAMA_API_KEY` when that endpoint requires a bearer token. Remote endpoints
must use HTTPS; HTTP is accepted only for loopback.

## Hugging Face Inference Providers

Provide a token through the host environment and select a model available to your account:

```sh
GABBY_RUN_HUGGINGFACE_INTEGRATION=1 \
GABBY_HUGGINGFACE_MODEL=org/model:provider \
HF_TOKEN=... \
uv run --frozen pytest tests/integration/test_model_providers_live.py -k huggingface -q
```

The adapter defaults to `https://router.huggingface.co/v1`; set
`GABBY_HUGGINGFACE_BASE_URL` to override it. Model availability depends on the selected model,
provider, account, and current service access. Do not put a real token in a command recorded to shell
history; use the host's secret manager or protected environment injection.

## Anthropic Messages API

Set an Anthropic API key through the host environment and choose a model enabled for that account:

```sh
GABBY_RUN_ANTHROPIC_INTEGRATION=1 \
GABBY_ANTHROPIC_MODEL=your-enabled-model-id \
uv run --frozen pytest tests/integration/test_model_providers_live.py -k anthropic -q
```

The check performs one completion and one streamed completion through the native Messages API. The
default endpoint is `https://api.anthropic.com/v1`; use `GABBY_ANTHROPIC_BASE_URL` for an approved
compatible endpoint. It can incur API usage charges. Keep `ANTHROPIC_API_KEY` in a secret manager or
protected environment injection and out of shell history. See Anthropic's
[Messages API reference](https://docs.anthropic.com/en/api/messages).

## Local Hugging Face Transformers

Install Gabby's `transformers` extra and a PyTorch 2.5+ build appropriate for the host's CPU or
accelerator. Point the test at a cached local checkpoint or a Hub model already available to the
host. Downloads are disabled by default for this acceptance command; set
`GABBY_TRANSFORMERS_LOCAL_FILES_ONLY=0` when you intend to download the configured model.

```sh
uv sync --extra transformers
GABBY_RUN_TRANSFORMERS_INTEGRATION=1 \
GABBY_TRANSFORMERS_MODEL=/path/to/local/chat-checkpoint \
GABBY_TRANSFORMERS_DEVICE=cpu \
uv run --frozen pytest tests/integration/test_model_providers_live.py -k local_transformers -q
```

Set `GABBY_TRANSFORMERS_DEVICE` to a supported PyTorch device such as `cuda` or `mps` when that
runtime is installed. `GABBY_TRANSFORMERS_REVISION` optionally pins a Hub commit. The acceptance
uses no tools and makes no claim about tool-template support, throughput, model quality, or
accelerator isolation; run a separate reviewed test with the selected model before relying on its
structured tool-call parser.

### Structured local tool-call parsing

This provider-level acceptance checks that a real model sees a tool schema and that its Transformers
tokenizer parses the response into Gabby's normalized tool-call contract. The model must emit one
tool call without unrelated assistant text in the same completion. This check does not measure tool
selection quality or prove that every model follows Gabby's multi-step runtime loop; deterministic
runtime tests cover dispatch and observation handling separately.

`tool_response_template` is model-specific. The schema below is for the pinned Qwen checkpoint and
should not be copied to another model without reviewing its chat and response formats.

```sh
HF_HOME=/tmp/gabby-hf-cache \
HF_HUB_CACHE=/tmp/gabby-hf-cache/hub \
GABBY_RUN_TRANSFORMERS_TOOL_INTEGRATION=1 \
GABBY_TRANSFORMERS_MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct \
GABBY_TRANSFORMERS_REVISION=ea3f2471cf1b1f0db85067f1ef93848e38e88c25 \
GABBY_TRANSFORMERS_LOCAL_FILES_ONLY=1 \
GABBY_TRANSFORMERS_DEVICE=cpu \
GABBY_TRANSFORMERS_TOOL_RESPONSE_TEMPLATE='{"start_anchor":"<|im_start|>assistant","fields":{"tool_calls":{"close":"<|im_end|>","content":"json","repeats":true}}}' \
uv run --frozen pytest tests/integration/test_model_providers_live.py -k structured_tool_call -q
```

The pinned CPU acceptance passed locally with Transformers 5.18.0 and PyTorch 2.14.1+cpu. The
checkpoint was already cached, so this command runs offline. To use another local or Hub model,
change the model and revision and provide the response schema its tokenizer expects.

A reproducible CPU completion smoke passed on 2026-10-02 with Transformers 5.18.0, PyTorch
2.14.1+cpu, and [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
at revision `eba0825ddc7730d24a88685a6c0a078cf919b383`. The model card identifies the checkpoint as
Apache-2.0 and documents its Transformers chat-template path. This verifies one model load and
completion on CPU; it does not verify tool calling or output quality. The same acceptance also
passed a second time with `GABBY_TRANSFORMERS_LOCAL_FILES_ONLY=1` against the cached checkpoint.

To reproduce the pinned run on a CPU host:

```sh
uv sync --frozen --group dev --extra transformers
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.14.1+cpu'
HF_HOME=/tmp/gabby-hf-cache \
HF_HUB_CACHE=/tmp/gabby-hf-cache/hub \
GABBY_RUN_TRANSFORMERS_INTEGRATION=1 \
GABBY_TRANSFORMERS_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct \
GABBY_TRANSFORMERS_REVISION=eba0825ddc7730d24a88685a6c0a078cf919b383 \
GABBY_TRANSFORMERS_LOCAL_FILES_ONLY=0 \
GABBY_TRANSFORMERS_DEVICE=cpu \
uv run --no-sync pytest tests/integration/test_model_providers_live.py -k local_transformers -q
```

## Local Transformers embeddings

This acceptance loads a model locally, batches two texts through `TransformersEmbeddingProvider`,
and checks vector count, dimensions, finite values, and default unit normalization. It does not
measure semantic quality or retrieval quality. Downloads are disabled by default:

If the pinned checkpoint is not in this cache yet, run the preceding pinned CPU completion setup
once with `GABBY_TRANSFORMERS_LOCAL_FILES_ONLY=0`; both acceptance checks use the same model and
cache directory. The embedding command below then loads it offline.

```sh
HF_HOME=/tmp/gabby-hf-cache \
HF_HUB_CACHE=/tmp/gabby-hf-cache/hub \
GABBY_RUN_TRANSFORMERS_EMBEDDINGS_INTEGRATION=1 \
GABBY_TRANSFORMERS_EMBEDDINGS_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct \
GABBY_TRANSFORMERS_EMBEDDINGS_REVISION=eba0825ddc7730d24a88685a6c0a078cf919b383 \
GABBY_TRANSFORMERS_EMBEDDINGS_LOCAL_FILES_ONLY=1 \
GABBY_TRANSFORMERS_EMBEDDINGS_DEVICE=cpu \
uv run --frozen pytest tests/integration/test_transformers_embeddings_live.py -q
```

The pinned CPU smoke passed on 2026-10-02 with Transformers 5.18.0 and PyTorch 2.14.1+cpu using
`HuggingFaceTB/SmolLM2-135M-Instruct` at revision
`eba0825ddc7730d24a88685a6c0a078cf919b383` and local-only loading. This validates model loading,
pooling, and normalized finite output; it does not measure semantic embedding quality. Install the
`transformers` extra and a host-compatible PyTorch build first. Set
`GABBY_TRANSFORMERS_EMBEDDINGS_LOCAL_FILES_ONLY=0` only when you intend to download another model.
Use a checkpoint supported by standard `AutoModel` with safetensors weights; Gabby disables custom
remote code and requires safetensors loading.

## Both providers

Set both opt-in flags, both model IDs, and `HF_TOKEN`, then run the test file without `-k`. Only the
enabled providers make requests. The assertions check non-empty completion output, streamed text
deltas, and one completion event. They do not evaluate answer quality or prove tool-calling behavior.

## Local LoRA training

This opt-in check creates a tiny random GPT-2 model and tokenizer locally, trains one LoRA step from
a one-row assistant-masked JSONL dataset, writes a safetensors adapter, then loads that adapter with
`TransformersProvider`. It downloads no model weights and measures no adaptation quality. Install a
host-compatible PyTorch build first, followed by Gabby's `transformers` and `training` extras, then
run:

```sh
GABBY_RUN_TRAINING_INTEGRATION=1 \
uv run --no-sync pytest tests/integration/test_local_lora_training_live.py -q
```

The test normally skips so the standard suite does not import the optional ML stack. The CPU
round-trip passed locally on 2026-10-03 with PyTorch 2.14.1+cpu, Transformers 5.18.0, PEFT 0.21.2,
and Accelerate 1.15.0. It verifies the installed stack for one train, load, and `Agent.arun()` cycle;
it does not measure adaptation quality. The deterministic contract tests remain routine CI coverage.
