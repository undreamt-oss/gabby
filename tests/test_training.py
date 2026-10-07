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
"""Bounded dataset and trainer contracts for optional LoRA SFT."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from gabby import training as training_module
from gabby.config import ConfigError
from gabby.training import (
    SFTConfig,
    TokenizedSFTExample,
    load_sft_dataset,
    tokenize_sft_examples,
    train_lora_sft,
)


def _row(*, messages: list[dict[str, str]] | None = None) -> dict[str, Any]:
    return {
        "messages": messages
        or [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "4"},
        ]
    }


def test_load_sft_dataset_validates_jsonl_and_returns_digest(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")

    examples, digest = load_sft_dataset(dataset)

    assert len(examples) == 2
    assert examples[0].messages[-1] == {"role": "assistant", "content": "4"}
    assert digest == hashlib.sha256(dataset.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "content",
    [
        '{"messages":[],"messages":[]}',
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}),
        json.dumps(
            {"messages": [{"role": "user", "content": "hi"}, {"role": "tool", "content": "x"}]}
        ),
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "ok"},
                ],
                "extra": True,
            }
        ),
        "not json",
    ],
)
def test_load_sft_dataset_rejects_invalid_rows(tmp_path: Path, content: str) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(content + "\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_sft_dataset(dataset)


def test_load_sft_dataset_enforces_example_and_file_bounds(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_row()) + "\n" + json.dumps(_row()) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="example limit"):
        load_sft_dataset(dataset, max_examples=1)
    with pytest.raises(ConfigError, match="byte limit"):
        load_sft_dataset(dataset, max_bytes=10)


def test_load_sft_dataset_rejects_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    real.write_text(json.dumps(_row()), encoding="utf-8")
    link = tmp_path / "link.jsonl"
    link.symlink_to(real)

    with pytest.raises(ConfigError, match="regular file"):
        load_sft_dataset(link)


def test_tokenize_sft_examples_masks_all_non_assistant_tokens() -> None:
    examples, _digest = load_sft_dataset_from_text()

    class Tokenizer:
        def apply_chat_template(self, conversation: Any, /, **kwargs: Any) -> dict[str, list[int]]:
            del conversation
            assert kwargs["return_assistant_tokens_mask"] is True
            assert kwargs["add_generation_prompt"] is False
            return {"input_ids": [10, 11, 12, 13], "assistant_masks": [0, 0, 1, 1]}

    tokenized = tokenize_sft_examples(examples, Tokenizer(), max_sequence_length=8)

    assert tokenized == (TokenizedSFTExample((10, 11, 12, 13), (-100, -100, 12, 13)),)


def test_tokenize_sft_examples_rejects_templates_without_assistant_masks() -> None:
    examples, _digest = load_sft_dataset_from_text()

    class Tokenizer:
        def apply_chat_template(self, conversation: Any, /, **_kwargs: Any) -> dict[str, list[int]]:
            del conversation
            return {"input_ids": [1, 2, 3], "assistant_masks": [0, 0, 0]}

    with pytest.raises(ConfigError, match="{% generation %}"):
        tokenize_sft_examples(examples, Tokenizer(), max_sequence_length=8)


def test_tokenize_sft_examples_rejects_oversized_sequences() -> None:
    examples, _digest = load_sft_dataset_from_text()

    class Tokenizer:
        def apply_chat_template(self, conversation: Any, /, **_kwargs: Any) -> dict[str, list[int]]:
            del conversation
            return {"input_ids": list(range(9)), "assistant_masks": [0, 0, 0, 0, 0, 0, 0, 1, 1]}

    with pytest.raises(ConfigError, match="exceeding max_sequence_length"):
        tokenize_sft_examples(examples, Tokenizer(), max_sequence_length=8)


def test_sft_config_rejects_invalid_bounds(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="lora_rank"):
        SFTConfig("model", tmp_path / "data.jsonl", tmp_path / "out", lora_rank=0)
    with pytest.raises(ConfigError, match="token_env"):
        SFTConfig("model", tmp_path / "data.jsonl", tmp_path / "out", token_env="BAD NAME")
    with pytest.raises(ConfigError, match="target_modules"):
        SFTConfig(
            "model",
            tmp_path / "data.jsonl",
            tmp_path / "out",
            target_modules=("q_proj", "q_proj"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", " ", "model_id"),
        ("model_id", None, "model_id"),
        ("revision", " ", "revision"),
        ("revision", 1, "revision"),
        ("token_env", "BAD NAME", "token_env"),
        ("token_env", None, "token_env"),
        ("local_files_only", 1, "local_files_only"),
        ("max_examples", True, "max_examples"),
        ("max_examples", 10_001, "max_examples"),
        ("max_sequence_length", 7, "max_sequence_length"),
        ("max_sequence_length", True, "max_sequence_length"),
        ("epochs", float("nan"), "epochs"),
        ("epochs", 0, "epochs"),
        ("batch_size", True, "batch_size"),
        ("gradient_accumulation_steps", 0, "gradient_accumulation_steps"),
        ("seed", -1, "seed"),
        ("learning_rate", float("inf"), "learning_rate"),
        ("learning_rate", 0, "learning_rate"),
        ("lora_rank", 257, "lora_rank"),
        ("lora_alpha", 0, "lora_alpha"),
        ("lora_dropout", 1, "lora_dropout"),
        ("lora_dropout", -0.1, "lora_dropout"),
        ("dataset", "train.jsonl", "pathlib.Path"),
        ("target_modules", ["q_proj"], "target_modules"),
        ("target_modules", ("",), "target_modules"),
        ("target_modules", tuple(f"module_{i}" for i in range(33)), "target_modules"),
    ],
)
def test_sft_config_rejects_invalid_field_types_and_edges(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        if field == "model_id":
            SFTConfig(value, tmp_path / "data.jsonl", tmp_path / "out")
        elif field == "dataset":
            SFTConfig("model", value, tmp_path / "out")
        else:
            SFTConfig("model", tmp_path / "data.jsonl", tmp_path / "out", **{field: value})


@pytest.mark.parametrize(
    ("max_examples", "max_bytes"),
    [
        (True, 10),
        (0, 10),
        (training_module.MAX_TRAINING_EXAMPLES + 1, 10),
        (1, True),
        (1, 0),
        (1, training_module.MAX_TRAINING_DATASET_BYTES + 1),
    ],
)
def test_load_sft_dataset_rejects_invalid_limits(
    tmp_path: Path, max_examples: int, max_bytes: int
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_row()), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sft_dataset(dataset, max_examples=max_examples, max_bytes=max_bytes)


@pytest.mark.parametrize(
    "content",
    [
        "",
        " \n\t\n",
        "\xff",
        "{" + "[" * 1200 + "]" * 1200,
        "x" * (training_module.MAX_TRAINING_LINE_BYTES + 1),
    ],
)
def test_load_sft_dataset_rejects_empty_undecodable_and_oversized_input(
    tmp_path: Path, content: str
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_bytes(content.encode("latin-1"))
    with pytest.raises(ConfigError):
        load_sft_dataset(dataset)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "user", "content": "hello"}],
        [
            {"role": "user", "content": "hello", "extra": True},
            {"role": "assistant", "content": "ok"},
        ],
        ["not a message", {"role": "assistant", "content": "ok"}],
        [{"role": "developer", "content": "hello"}, {"role": "assistant", "content": "ok"}],
        [{"role": "user", "content": "  "}, {"role": "assistant", "content": "ok"}],
        [{"role": "user", "content": 1}, {"role": "assistant", "content": "ok"}],
        [{"role": "assistant", "content": "ok"}, {"role": "assistant", "content": "ok"}],
        [{"role": "user", "content": "hello"}, {"role": "user", "content": "again"}],
        [{"role": "user", "content": "\ud800"}, {"role": "assistant", "content": "ok"}],
        [
            {"role": "user", "content": "x" * (training_module.MAX_TRAINING_MESSAGE_BYTES + 1)},
            {"role": "assistant", "content": "ok"},
        ],
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "ok"},
            *(
                [
                    {"role": "user", "content": "extra"},
                    {"role": "assistant", "content": "extra"},
                ]
                * (training_module.MAX_TRAINING_MESSAGES // 2)
            ),
        ],
    ],
)
def test_load_sft_dataset_rejects_invalid_message_contracts(
    tmp_path: Path, messages: list[dict[str, Any]]
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps({"messages": messages}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_sft_dataset(dataset)


def test_tokenize_sft_examples_sanitizes_template_failures_and_bad_encodings() -> None:
    examples, _digest = load_sft_dataset_from_text()

    class RaisingTokenizer:
        def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("private template details")

    with pytest.raises(ConfigError, match="chat template failed") as failure:
        tokenize_sft_examples(examples, RaisingTokenizer(), max_sequence_length=8)
    assert "private template details" not in str(failure.value)

    class InvalidTokenizer:
        def __init__(self, result: Any) -> None:
            self.result = result

        def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> Any:
            return self.result

    for result in (
        None,
        {"input_ids": [1, 2], "assistant_masks": [0]},
        {"input_ids": [1, 2], "assistant_masks": [0, 2]},
        {"input_ids": [1, True], "assistant_masks": [0, 1]},
    ):
        with pytest.raises(ConfigError):
            tokenize_sft_examples(examples, InvalidTokenizer(result), max_sequence_length=8)


def test_tokenize_sft_examples_enforces_total_token_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    examples, _digest = load_sft_dataset_from_text()
    monkeypatch.setattr(training_module, "MAX_TOTAL_TRAINING_TOKENS", 3)

    class Tokenizer:
        def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> dict[str, list[int]]:
            return {"input_ids": [1, 2], "assistant_masks": [1, 1]}

    with pytest.raises(ConfigError, match="total limit"):
        tokenize_sft_examples(examples * 2, Tokenizer(), max_sequence_length=8)


def test_load_sft_dataset_enforces_record_line_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("\n\n", encoding="utf-8")
    monkeypatch.setattr(training_module, "MAX_TRAINING_RECORD_LINES", 1)
    with pytest.raises(ConfigError, match="line limit"):
        load_sft_dataset(dataset)


def test_load_sft_dataset_sanitizes_open_race_and_read_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_row()), encoding="utf-8")

    def changed_file(_descriptor: int) -> Any:
        return SimpleNamespace(st_mode=0o100600, st_dev=1, st_ino=2)

    monkeypatch.setattr(training_module.os, "fstat", changed_file)
    with pytest.raises(ConfigError, match="changed while"):
        load_sft_dataset(dataset)

    monkeypatch.undo()
    monkeypatch.setattr(
        training_module.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private path detail")),
    )
    with pytest.raises(ConfigError, match="could not be opened") as failure:
        load_sft_dataset(dataset)
    assert "private path detail" not in str(failure.value)


def test_load_sft_dataset_sanitizes_read_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_row()), encoding="utf-8")

    class FailedStream:
        def __enter__(self) -> FailedStream:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def readline(self, _limit: int) -> bytes:
            raise OSError("private read detail")

    def failed_fdopen(descriptor: int, _mode: str) -> FailedStream:
        os.close(descriptor)
        return FailedStream()

    monkeypatch.setattr(training_module.os, "fdopen", failed_fdopen)
    with pytest.raises(ConfigError, match="could not be read") as failure:
        load_sft_dataset(dataset)
    assert "private read detail" not in str(failure.value)


def test_train_lora_sft_rejects_existing_output_before_optional_imports(tmp_path: Path) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_row()), encoding="utf-8")
    output = tmp_path / "adapter"
    output.mkdir()
    with pytest.raises(ConfigError, match="already exists"):
        train_lora_sft(SFTConfig("model", dataset, output))


def test_training_dataset_and_collator_preserve_assistant_only_padding() -> None:
    class FakeTensor(list[int]):
        @property
        def shape(self) -> tuple[int]:
            return (len(self),)

    class Matrix:
        def __init__(self, rows: list[list[int]]) -> None:
            self.rows = rows

        def __setitem__(self, key: tuple[int, slice], value: Any) -> None:
            row, selected = key
            if isinstance(value, int):
                value = [value] * len(range(*selected.indices(len(self.rows[row]))))
            self.rows[row][selected] = value

    class FakeTorch:
        long = "long"

        @staticmethod
        def tensor(values: Any, **_kwargs: Any) -> FakeTensor:
            return FakeTensor(values)

        @staticmethod
        def ones(length: int, **_kwargs: Any) -> FakeTensor:
            return FakeTensor([1] * length)

        @staticmethod
        def full(shape: tuple[int, int], value: int, **_kwargs: Any) -> Matrix:
            return Matrix([[value] * shape[1] for _ in range(shape[0])])

        @staticmethod
        def zeros(shape: tuple[int, int], **_kwargs: Any) -> Matrix:
            return Matrix([[0] * shape[1] for _ in range(shape[0])])

    examples = (
        TokenizedSFTExample((1, 2, 3), (-100, 2, 3)),
        TokenizedSFTExample((4, 5), (-100, 5)),
    )
    dataset = training_module._TorchSFTDataset(FakeTorch, examples)
    assert len(dataset) == 2
    features = [dataset[0], dataset[1]]
    batch = training_module._AssistantOnlyCollator(FakeTorch, pad_token_id=0)(features)
    assert batch["input_ids"].rows == [[1, 2, 3], [4, 5, 0]]
    assert batch["labels"].rows == [[-100, 2, 3], [-100, 5, -100]]
    assert batch["attention_mask"].rows == [[1, 1, 1], [1, 1, 0]]


def test_training_helpers_keep_metrics_bounded_and_manifests_exclusive(tmp_path: Path) -> None:
    assert training_module._int_list(SimpleNamespace(tolist=lambda: [[1, 2]])) == [1, 2]
    assert training_module._int_list([1, True]) == []
    assert training_module._safe_metrics(
        {"loss": 0.25, "step": 2, "enabled": True, "infinite": float("inf"), 3: 4}
    ) == {"loss": 0.25, "step": 2, "enabled": "True"}
    assert training_module._safe_metrics(None) == {}

    versions = training_module._package_versions()
    assert set(versions) == {"gabby-agent-runtime", "transformers", "peft", "accelerate", "torch"}
    assert all(value is None or isinstance(value, str) for value in versions.values())

    manifest_path = tmp_path / "manifest.json"
    training_module._write_manifest(manifest_path, {"format": "test"})
    assert manifest_path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        training_module._write_manifest(manifest_path, {"format": "replacement"})


def test_train_lora_sft_sanitizes_missing_optional_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(json.dumps(_row()), encoding="utf-8")
    output = tmp_path / "adapter"
    config = SFTConfig("model", dataset, output)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ConfigError, match="platform-selected PyTorch"):
        train_lora_sft(config)
    assert not output.exists()
    assert list(tmp_path.glob(".*.gabby-training-*")) == []


def test_train_lora_sft_writes_only_adapter_and_provenance_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_path = tmp_path / "train.jsonl"
    dataset_path.write_text(json.dumps(_row()), encoding="utf-8")
    output = tmp_path / "adapter"
    tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=0,
        eos_token="<eos>",
        pad_token="<pad>",
        apply_chat_template=lambda *_args, **_kwargs: {
            "input_ids": [1, 2, 3, 4],
            "assistant_masks": [0, 0, 1, 1],
        },
        save_pretrained=lambda path: (Path(path) / "tokenizer.json").write_text("{}"),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(pad_token_id=None),
        save_pretrained=lambda path, **_kwargs: (Path(path) / "adapter.safetensors").write_bytes(
            b"weights"
        ),
    )
    torch_module = ModuleType("torch")
    torch_module.__dict__["long"] = "long"
    torch_module.__dict__["cuda"] = SimpleNamespace(is_available=lambda: False)
    torch_module.__dict__["tensor"] = lambda values, **_kwargs: tuple(values)
    torch_module.__dict__["ones"] = lambda length, **_kwargs: tuple(1 for _ in range(length))
    torch_module.__dict__["full"] = lambda shape, value, **_kwargs: tuple(
        tuple(value for _ in range(shape[1])) for _ in range(shape[0])
    )
    transformers_module = ModuleType("transformers")
    transformers_module.__dict__.update(
        {
            "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: tokenizer),
            "AutoModelForCausalLM": SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: model
            ),
            "TrainingArguments": lambda **kwargs: SimpleNamespace(**kwargs),
        }
    )

    class FakeTrainer:
        def __init__(
            self, *, model: Any, train_dataset: Any, data_collator: Any, **_kwargs: Any
        ) -> None:
            self.model = model
            self.train_dataset = train_dataset
            self.data_collator = data_collator

        def train(self) -> SimpleNamespace:
            item = self.train_dataset[0]
            assert item["labels"] == (-100, -100, 3, 4)
            return SimpleNamespace(metrics={"train_loss": 0.25, "global_step": 1})

    transformers_module.__dict__["Trainer"] = FakeTrainer
    peft_module = ModuleType("peft")
    peft_module.__dict__.update(
        {
            "LoraConfig": lambda **kwargs: kwargs,
            "TaskType": SimpleNamespace(CAUSAL_LM="CAUSAL_LM"),
            "get_peft_model": lambda base, _config: base,
        }
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)

    manifest = train_lora_sft(
        SFTConfig(
            "org/base",
            dataset_path,
            output,
            revision="base-sha",
            target_modules=("q_proj", "v_proj"),
        )
    )

    assert manifest["base_revision"] == "base-sha"
    assert manifest["example_count"] == 1
    assert manifest["metrics"] == {"train_loss": 0.25, "global_step": 1}
    assert (output / "adapter.safetensors").is_file()
    assert (output / "gabby-training.json").is_file()
    assert "What is 2 + 2?" not in (output / "gabby-training.json").read_text(encoding="utf-8")
    assert list(tmp_path.glob(".*.gabby-training-*")) == []


def load_sft_dataset_from_text() -> tuple[Any, str]:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        path.write_text(json.dumps(_row()), encoding="utf-8")
        return load_sft_dataset(path)
