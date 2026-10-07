# Local adapter training

Gabby separates runtime specialization from model-weight adaptation. Instructions, skills, tools,
knowledge retrieval, and policies specialize an agent without changing the model. The optional
`gabby train` command performs local supervised fine-tuning (SFT) of a PEFT LoRA adapter. The
training command does not run inside an agent request and does not upload the dataset.

## Install

Install a PyTorch build that matches the machine first, using the
[official PyTorch selector](https://pytorch.org/get-started/locally/). In an existing environment,
then install Gabby's Transformers and training extras with `uv pip` or `pip`:

```bash
uv pip install 'gabby-agent-runtime[transformers,training]'
```

The extras provide Transformers, PEFT, and Accelerate. PyTorch is selected separately because its
CPU and accelerator builds vary by host. From a Gabby checkout, use
`uv pip install -e '.[transformers,training]'`. Install the selected PyTorch build before these
extras so the resolver reuses it. `uv sync --extra training` follows the lockfile's default PyPI
PyTorch build and may not match a host's accelerator choice. Training downloads base model files from
the configured Hugging Face model ID unless the model is already cached.

## Dataset format

Provide UTF-8 JSONL with one conversation per line. Each record contains only a `messages` array;
messages have `role` and `content` string fields. Roles are `system`, `user`, and `assistant`. Every
record must include a user message and end with an assistant message. Tool-call traces and multimodal
content are not accepted by this initial trainer.

```jsonl
{"messages":[{"role":"user","content":"What is 17 plus 25?"},{"role":"assistant","content":"42"}]}
{"messages":[{"role":"system","content":"Answer with a short explanation."},{"role":"user","content":"Why does ice float?"},{"role":"assistant","content":"Ice is less dense than liquid water because its crystal structure spaces the molecules farther apart."}]}
```

Gabby bounds a dataset to 100 MiB, 10,000 examples, 20,000 physical lines, 1 MiB per JSONL line,
64 messages per example, 64 KiB per message, 32,768 tokens per example, and 4 million total
tokenized tokens. Records with duplicate JSON keys, invalid Unicode, unsupported roles, or malformed
messages are rejected. Each tokenizer chat template must return an assistant-token mask, which
requires the template to mark generated assistant spans using `{% generation %}`. The trainer
rejects templates that cannot identify assistant tokens rather than training on user and system
text.

## Train a LoRA adapter

Start with a small, representative dataset and a model whose chat template supports assistant masks.
Pin the base revision to a commit when reproducibility matters:

```bash
gabby train \
  --model HuggingFaceTB/SmolLM3-3B \
  --revision <base-model-commit> \
  --dataset ./training.jsonl \
  --output ./artifacts/support-adapter \
  --target-module q_proj \
  --target-module v_proj
```

Validate the JSONL structure and get its digest before installing training dependencies:

```bash
gabby train --dataset ./training.jsonl --check-dataset
```

This checks record and resource bounds; it does not load a tokenizer or prove that the chosen model's
chat template supports assistant-only loss. The actual training run checks that mask after loading
the tokenizer.

Output paths must not already exist; artifacts are written to a staging directory and published only
after successful training. Defaults are one epoch, batch size 1, gradient accumulation 8, seed 42,
learning rate `0.0002`, LoRA rank 8, alpha 16, dropout `0.05`, and sequence length 2,048. These are
starting values, not universal recommendations. Model size, target modules, dataset quality, and
hardware determine whether training fits and produces a useful adapter. Training runs locally on the
host and can consume substantial memory, accelerator time, and disk space.

The command saves PEFT adapter files in safetensors format and writes `gabby-training.json` with the
base model and revision, local-only setting, Python and training package versions, SHA-256 digest of
the source JSONL, example count, random seed, training settings, and finite training metrics. It never stores
training message text in the manifest. `--local-files-only` prevents model downloads. The digest supports
dataset identity checks; it does not prove data rights, quality, or privacy. The base model and
adapter licenses and provenance remain the operator's responsibility.

## Use the adapter

Point a local Transformers agent at the base model and generated adapter directory:

```yaml
model:
  provider: transformers
  model: HuggingFaceTB/SmolLM3-3B
  revision: <base-model-commit>
  adapter_id: ./artifacts/support-adapter
  device: cpu
```

Install `gabby-agent-runtime[transformers,transformers-adapters]` for inference. For a Hub-hosted
adapter, use its repository ID and optionally pin `adapter_revision`; private Hub access uses the
host's `HF_TOKEN` environment variable. Base-model instructions, tools, knowledge, and policies still
define runtime behavior and security. Fine-tuning does not replace those controls, and Gabby does not
currently provide training-set evaluation, hyperparameter search, QLoRA, distributed training, or
adapter publishing. Evaluate the generated agent on held-out tasks with `gabby evaluate` before
deployment.
