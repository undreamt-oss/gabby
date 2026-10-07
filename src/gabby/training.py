# Copyright 2026-present Gabby Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional, bounded local supervised fine-tuning for PEFT adapters."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from .config import ConfigError

MAX_TRAINING_DATASET_BYTES = 100 * 1024 * 1024
MAX_TRAINING_EXAMPLES = 10_000
MAX_TRAINING_LINE_BYTES = 1024 * 1024
MAX_TRAINING_MESSAGES = 64
MAX_TRAINING_MESSAGE_BYTES = 64 * 1024
MAX_TRAINING_SEQUENCE_LENGTH = 32_768
MAX_TOTAL_TRAINING_TOKENS = 4_000_000
MAX_TRAINING_RECORD_LINES = MAX_TRAINING_EXAMPLES * 2


@dataclass(frozen=True, slots=True)
class SFTConfig:
    """Bounded configuration for local LoRA supervised fine-tuning."""

    model_id: str
    dataset: Path
    output: Path
    revision: str | None = None
    token_env: str = "HF_TOKEN"
    local_files_only: bool = False
    max_examples: int = 10_000
    max_sequence_length: int = 2048
    epochs: float = 1.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    seed: int = 42
    learning_rate: float = 0.0002
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject invalid bounds before importing optional ML frameworks."""
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ConfigError("Training model_id must be a non-empty model ID or local path")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ConfigError("Training revision must be a non-empty string or None")
        if not isinstance(self.token_env, str) or not self.token_env.isidentifier():
            raise ConfigError("Training token_env must be an environment variable name")
        if not isinstance(self.local_files_only, bool):
            raise ConfigError("local_files_only must be a boolean")
        if (
            isinstance(self.max_examples, bool)
            or not isinstance(self.max_examples, int)
            or not 1 <= self.max_examples <= MAX_TRAINING_EXAMPLES
        ):
            raise ConfigError(f"max_examples must be from 1 through {MAX_TRAINING_EXAMPLES}")
        if (
            isinstance(self.max_sequence_length, bool)
            or not isinstance(self.max_sequence_length, int)
            or not 8 <= self.max_sequence_length <= MAX_TRAINING_SEQUENCE_LENGTH
        ):
            raise ConfigError(
                f"max_sequence_length must be from 8 through {MAX_TRAINING_SEQUENCE_LENGTH}"
            )
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, (float, int))
            or not math.isfinite(self.epochs)
            or not 0 < self.epochs <= 100
        ):
            raise ConfigError("epochs must be finite and greater than 0, up to 100")
        for name, value, upper in (
            ("batch_size", self.batch_size, 64),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps, 1024),
            ("lora_rank", self.lora_rank, 256),
            ("lora_alpha", self.lora_alpha, 1024),
            ("seed", self.seed, 2**32 - 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ConfigError(f"{name} must be an integer from 1 through {upper}")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (float, int))
            or not math.isfinite(self.learning_rate)
            or not 1e-8 <= self.learning_rate <= 1
        ):
            raise ConfigError("learning_rate must be finite and from 1e-8 through 1")
        if (
            isinstance(self.lora_dropout, bool)
            or not isinstance(self.lora_dropout, (float, int))
            or not math.isfinite(self.lora_dropout)
            or not 0 <= self.lora_dropout < 1
        ):
            raise ConfigError("lora_dropout must be finite and from 0 up to (but not including) 1")
        if not isinstance(self.dataset, Path) or not isinstance(self.output, Path):
            raise ConfigError("dataset and output must be pathlib.Path values")
        if (
            not isinstance(self.target_modules, tuple)
            or len(self.target_modules) > 32
            or any(
                not isinstance(name, str) or not name.strip() or len(name) > 128
                for name in self.target_modules
            )
            or len(set(self.target_modules)) != len(self.target_modules)
        ):
            raise ConfigError("target_modules must contain up to 32 unique non-empty names")


@dataclass(frozen=True, slots=True)
class SFTExample:
    """Validated conversation training row."""

    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class TokenizedSFTExample:
    """One sequence with labels enabled only for assistant-generated tokens."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


class ChatTokenizer(Protocol):
    """Subset of the Hugging Face tokenizer contract used during dataset preparation."""

    def apply_chat_template(self, conversation: Any, /, **kwargs: Any) -> Any: ...


def load_sft_dataset(
    path: Path,
    *,
    max_examples: int = MAX_TRAINING_EXAMPLES,
    max_bytes: int = MAX_TRAINING_DATASET_BYTES,
) -> tuple[tuple[SFTExample, ...], str]:
    """Load bounded UTF-8 JSONL conversations and return their SHA-256 digest."""
    if isinstance(max_examples, bool) or not isinstance(max_examples, int):
        raise ConfigError("max_examples must be an integer")
    if not 1 <= max_examples <= MAX_TRAINING_EXAMPLES:
        raise ConfigError(f"max_examples must be from 1 through {MAX_TRAINING_EXAMPLES}")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ConfigError("max_bytes must be a positive integer")
    if max_bytes > MAX_TRAINING_DATASET_BYTES:
        raise ConfigError(f"max_bytes cannot exceed {MAX_TRAINING_DATASET_BYTES}")
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ConfigError("Training dataset must be a regular file, not a symlink")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            metadata.st_dev,
            metadata.st_ino,
        ):
            os.close(descriptor)
            raise ConfigError("Training dataset changed while it was being opened")
        stream = os.fdopen(descriptor, "rb")
    except ConfigError:
        raise
    except OSError:
        raise ConfigError("Training dataset could not be opened as a regular file") from None

    examples: list[SFTExample] = []
    digest = hashlib.sha256()
    total_bytes = 0
    try:
        with stream:
            line_number = 0
            while raw_line := stream.readline(MAX_TRAINING_LINE_BYTES + 1):
                line_number += 1
                if line_number > MAX_TRAINING_RECORD_LINES:
                    raise ConfigError(
                        f"Training dataset exceeds the {MAX_TRAINING_RECORD_LINES}-line limit"
                    )
                total_bytes += len(raw_line)
                if total_bytes > max_bytes:
                    raise ConfigError(f"Training dataset exceeds the {max_bytes}-byte limit")
                if len(raw_line) > MAX_TRAINING_LINE_BYTES:
                    raise ConfigError(
                        f"Training dataset line {line_number} exceeds "
                        f"{MAX_TRAINING_LINE_BYTES} bytes"
                    )
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"), object_pairs_hook=_unique_object)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
                    raise ConfigError(
                        f"Training dataset line {line_number} is not valid JSONL"
                    ) from None
                if not isinstance(row, dict) or set(row) != {"messages"}:
                    raise ConfigError(
                        f"Training dataset line {line_number} must contain only a messages array"
                    )
                messages = _validate_messages(row["messages"], line_number)
                examples.append(SFTExample(messages=messages))
                if len(examples) > max_examples:
                    raise ConfigError(f"Training dataset exceeds the {max_examples}-example limit")
    except OSError:
        raise ConfigError("Training dataset could not be read") from None
    if not examples:
        raise ConfigError("Training dataset must contain at least one non-empty JSONL record")
    return tuple(examples), digest.hexdigest()


def tokenize_sft_examples(
    examples: tuple[SFTExample, ...],
    tokenizer: ChatTokenizer,
    *,
    max_sequence_length: int,
) -> tuple[TokenizedSFTExample, ...]:
    """Use model chat templates and train loss only on explicitly marked assistant tokens."""
    if not 8 <= max_sequence_length <= MAX_TRAINING_SEQUENCE_LENGTH:
        raise ConfigError(
            f"max_sequence_length must be from 8 through {MAX_TRAINING_SEQUENCE_LENGTH}"
        )
    prepared: list[TokenizedSFTExample] = []
    total_tokens = 0
    for index, example in enumerate(examples, start=1):
        try:
            encoded = tokenizer.apply_chat_template(
                list(example.messages),
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
                add_generation_prompt=False,
            )
        except Exception:
            raise ConfigError(
                f"Model chat template failed while encoding training example {index}"
            ) from None
        if not isinstance(encoded, Mapping):
            raise ConfigError(
                f"Model chat template returned an invalid encoding for training example {index}"
            )
        input_ids = _int_list(encoded.get("input_ids"))
        assistant_mask = _int_list(encoded.get("assistant_masks"))
        if len(input_ids) != len(assistant_mask) or not input_ids:
            raise ConfigError(
                "Model chat template returned an invalid assistant mask for training "
                f"example {index}"
            )
        if any(value not in (0, 1) for value in assistant_mask):
            raise ConfigError(
                f"Model chat template returned invalid assistant mask values for example {index}"
            )
        if len(input_ids) > max_sequence_length:
            raise ConfigError(
                f"Training example {index} has {len(input_ids)} tokens, exceeding "
                f"max_sequence_length={max_sequence_length}"
            )
        total_tokens += len(input_ids)
        if total_tokens > MAX_TOTAL_TRAINING_TOKENS:
            raise ConfigError(
                f"Training dataset exceeds the {MAX_TOTAL_TRAINING_TOKENS}-token total limit"
            )
        labels = tuple(
            token if mask else -100 for token, mask in zip(input_ids, assistant_mask, strict=True)
        )
        if sum(mask != 0 for mask in assistant_mask) < 2:
            raise ConfigError(
                f"Training example {index} has no usable assistant-token mask; "
                "the chat template must mark assistant spans with {% generation %}"
            )
        prepared.append(TokenizedSFTExample(input_ids=tuple(input_ids), labels=labels))
    return tuple(prepared)


def train_lora_sft(config: SFTConfig) -> dict[str, Any]:
    """Train a local LoRA adapter with bounded assistant-only supervised examples."""
    examples, dataset_sha256 = load_sft_dataset(config.dataset, max_examples=config.max_examples)
    if config.output.exists() or config.output.is_symlink():
        raise ConfigError(f"Training output already exists: {config.output}")
    try:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        staging_output = Path(
            tempfile.mkdtemp(
                prefix=f".{config.output.name}.gabby-training-", dir=config.output.parent
            )
        )
    except OSError:
        raise ConfigError("Training output directory could not be created") from None

    token = os.environ.get(config.token_env)
    load_options: dict[str, Any] = {
        "trust_remote_code": False,
        "use_safetensors": True,
        "local_files_only": config.local_files_only,
    }
    if token:
        load_options["token"] = token
    if config.revision:
        load_options["revision"] = config.revision
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ImportError:
        shutil.rmtree(staging_output, ignore_errors=True)
        raise ConfigError(
            "Install Gabby's optional 'transformers', 'training', and platform-selected "
            "PyTorch dependencies to train an adapter"
        ) from None

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **load_options)
        model: Any = AutoModelForCausalLM.from_pretrained(config.model_id, **load_options)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ConfigError(
                    "Training tokenizer must define a padding or end-of-sequence token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        tokenizer_contract = cast(ChatTokenizer, tokenizer)
        tokenized = tokenize_sft_examples(
            examples, tokenizer_contract, max_sequence_length=config.max_sequence_length
        )
        dataset = _TorchSFTDataset(torch, tokenized)
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=list(config.target_modules) or None,
            ),
        )
        training_args = TrainingArguments(
            output_dir=str(staging_output),
            num_train_epochs=float(config.epochs),
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            seed=config.seed,
            data_seed=config.seed,
            learning_rate=float(config.learning_rate),
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=10,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=bool(torch.cuda.is_available()),
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=_AssistantOnlyCollator(torch, tokenizer.pad_token_id),
        )
        trained = trainer.train()
        trained_model = trainer.model
        if trained_model is None:
            raise ConfigError("Training backend did not return the trained model")
        cast(Any, trained_model).save_pretrained(staging_output, safe_serialization=True)
        tokenizer.save_pretrained(staging_output)
        metrics = _safe_metrics(trained.metrics)
        manifest = {
            "format": "gabby-sft-lora-v1",
            "base_model": config.model_id,
            "base_revision": config.revision,
            "local_files_only": config.local_files_only,
            "python_version": platform.python_version(),
            "package_versions": _package_versions(),
            "dataset_sha256": dataset_sha256,
            "example_count": len(tokenized),
            "max_sequence_length": config.max_sequence_length,
            "training": {
                "epochs": float(config.epochs),
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "seed": config.seed,
                "learning_rate": float(config.learning_rate),
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": float(config.lora_dropout),
                "target_modules": list(config.target_modules),
            },
            "metrics": metrics,
        }
        _write_manifest(staging_output / "gabby-training.json", manifest)
        if config.output.exists() or config.output.is_symlink():
            raise ConfigError(f"Training output already exists: {config.output}")
        os.rename(staging_output, config.output)
        return manifest
    except ConfigError:
        shutil.rmtree(staging_output, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(staging_output, ignore_errors=True)
        raise ConfigError("LoRA training failed; model and dataset details were withheld") from None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _validate_messages(value: Any, line_number: int) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not 2 <= len(value) <= MAX_TRAINING_MESSAGES:
        raise ConfigError(
            f"Training dataset line {line_number} messages must contain 2 through "
            f"{MAX_TRAINING_MESSAGES} items"
        )
    messages: list[dict[str, str]] = []
    has_user = False
    for message in value:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ConfigError(
                f"Training dataset line {line_number} messages require role and content fields"
            )
        role = message["role"]
        content = message["content"]
        if not isinstance(role, str) or role not in {"system", "user", "assistant"}:
            raise ConfigError(
                f"Training dataset line {line_number} contains an unsupported message role"
            )
        if not isinstance(content, str) or not content.strip():
            raise ConfigError(
                f"Training dataset line {line_number} message content must be non-empty text"
            )
        try:
            content_size = len(content.encode("utf-8"))
        except UnicodeEncodeError:
            raise ConfigError(
                f"Training dataset line {line_number} message content is not valid Unicode"
            ) from None
        if content_size > MAX_TRAINING_MESSAGE_BYTES:
            raise ConfigError(
                f"Training dataset line {line_number} message exceeds "
                f"{MAX_TRAINING_MESSAGE_BYTES} bytes"
            )
        has_user = has_user or role == "user"
        messages.append({"role": role, "content": content})
    if not has_user or messages[-1]["role"] != "assistant":
        raise ConfigError(
            f"Training dataset line {line_number} must include a user message and "
            "end with assistant"
        )
    return tuple(messages)


def _int_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        return []
    return value


class _TorchSFTDataset:
    def __init__(self, torch: Any, examples: tuple[TokenizedSFTExample, ...]) -> None:
        self._torch = torch
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self._examples[index]
        return {
            "input_ids": self._torch.tensor(example.input_ids, dtype=self._torch.long),
            "labels": self._torch.tensor(example.labels, dtype=self._torch.long),
            "attention_mask": self._torch.ones(len(example.input_ids), dtype=self._torch.long),
        }


class _AssistantOnlyCollator:
    def __init__(self, torch: Any, pad_token_id: int) -> None:
        self._torch = torch
        self._pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        maximum = max(int(item["input_ids"].shape[0]) for item in features)
        batch = len(features)
        input_ids = self._torch.full((batch, maximum), self._pad_token_id, dtype=self._torch.long)
        labels = self._torch.full((batch, maximum), -100, dtype=self._torch.long)
        attention_mask = self._torch.zeros((batch, maximum), dtype=self._torch.long)
        for row, item in enumerate(features):
            length = int(item["input_ids"].shape[0])
            input_ids[row, :length] = item["input_ids"]
            labels[row, :length] = item["labels"]
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _safe_metrics(value: Any) -> dict[str, int | float | str]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, int | float | str] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(item, bool):
            metrics[key] = str(item)
        elif isinstance(item, int) or isinstance(item, float) and math.isfinite(item):
            metrics[key] = item
    return metrics


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("gabby-agent-runtime", "transformers", "peft", "accelerate", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    content = json.dumps(manifest, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
