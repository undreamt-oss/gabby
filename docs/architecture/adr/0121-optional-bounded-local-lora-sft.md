# ADR 0121: Optional bounded local LoRA SFT

- Status: accepted
- Date: 2026-10-03

## Context

Agent runtime composition specializes behavior without changing model weights. Some developers also
need to adapt a local model for a domain or response style and then use that adapter in a stateless
agent. This is a separate workflow from inference and must not add training dependencies or model
downloads to ordinary Gabby installations.

## Decision

Provide an optional `gabby train` command using local Hugging Face Transformers and PEFT to perform
supervised fine-tuning of a LoRA adapter. The command reads a bounded UTF-8 JSONL conversation set,
requires each example to contain a user message and end with an assistant message, uses the selected
tokenizer chat template, and enables loss only for assistant tokens explicitly marked by the
template's assistant mask. Unsupported or unmarked templates fail closed.

The initial input contract accepts text `system`, `user`, and `assistant` messages, caps the dataset
at 100 MiB and 10,000 examples, caps each example at 32,768 tokens, and caps the complete tokenized
dataset at four million tokens. Training parameters and LoRA rank are bounded. Base model loading
disables remote custom code and requires safetensors. Trained adapters are saved as safetensors into
a new output directory. A manifest records the base revision, source dataset SHA-256, example count,
training settings, and finite metrics, but not message contents.

`transformers` and `training` are optional extras; the operator installs the appropriate PyTorch
build for the host. Training stays an explicit local command and is never invoked by `Agent.run()`,
FastAPI, or a served request. The resulting adapter is loaded using the separate inference-only PEFT
configuration contract in [ADR 0120](0120-inference-only-peft-adapter-loading.md).

## Consequences

- A local developer can train an adapter and configure a local Transformers agent to use it.
- Agent runtime execution does not change, and ordinary providers do not import training libraries.
- The dataset is not sent to a Gabby service; downloading a remote base model still contacts its
  configured model host.
- Gabby does not provide QLoRA, distributed training, validation-set evaluation, hyperparameter
  search, or adapter publishing in this first workflow.
- Training quality, compute requirements, dataset rights, and model/adapter license compatibility
  remain the operator's responsibility.

## Alternatives considered

- Treat prompts, skills, or retrieval as training: rejected because they do not modify model weights.
- Require a host-defined training plugin only: rejected for the first local path because it would
  leave the optional local Transformers model path without a concrete adaptation workflow.
- Add hosted training to the runtime API: rejected because it would couple stateless request serving
  to long-running accelerator jobs, dataset persistence, quotas, and job lifecycle management.
- Use assistant-only loss for every tokenizer template: rejected because templates that cannot
  identify assistant spans would silently train on user and system text.
