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
"""Opt-in offline acceptance for SFT training and generated adapter inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from gabby.agent import Agent
from gabby.config import AgentDefinition
from gabby.models import TransformersProvider
from gabby.training import SFTConfig, train_lora_sft


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_lora_training_produces_adapter_usable_by_agent(tmp_path: Path) -> None:
    """Train a tiny random local model without contacting a model host."""
    run_flag = "GABBY_RUN_TRAINING_INTEGRATION"
    if os.environ.get(run_flag) != "1":
        pytest.skip(f"set {run_flag}=1 to run local LoRA training acceptance")
    try:
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    except ImportError:
        pytest.skip("install the transformers, training, and host-selected PyTorch dependencies")

    torch.manual_seed(7)
    torch.set_num_threads(1)
    vocabulary = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<eos>": 2,
        "<|system|>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
        "What": 6,
        "is": 7,
        "two": 8,
        "plus": 9,
        "?": 10,
        "Four": 11,
        "numbers": 12,
        "are": 13,
        "added.": 14,
    }
    tokenizer_constructor: Any = Tokenizer
    backend_tokenizer = tokenizer_constructor(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]  # Optional Transformers.
        tokenizer_object=backend_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<eos>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>' }} "
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }} {{ eos_token }}{% endgeneration %}"
        "{% else %}{{ message['content'] }} {% endif %}"
        "{% endfor %}"
    )
    base_model_path = tmp_path / "base-model"
    base_model_path.mkdir()
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(model_config)  # type: ignore[no-untyped-call]  # Optional Transformers.
    model.save_pretrained(base_model_path, safe_serialization=True)
    tokenizer.save_pretrained(base_model_path)
    dataset = tmp_path / "training.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is two plus two?"},
                    {"role": "assistant", "content": "Four numbers are added."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter_path = tmp_path / "trained-adapter"

    manifest = train_lora_sft(
        SFTConfig(
            model_id=str(base_model_path),
            dataset=dataset,
            output=adapter_path,
            local_files_only=True,
            max_sequence_length=32,
            epochs=1,
            batch_size=1,
            gradient_accumulation_steps=1,
            lora_rank=2,
            lora_alpha=4,
            target_modules=("c_attn",),
        )
    )

    assert manifest["example_count"] == 1
    assert (adapter_path / "adapter_model.safetensors").is_file()
    provider = TransformersProvider(
        model_id=str(base_model_path),
        adapter_id=str(adapter_path),
        local_files_only=True,
        max_new_tokens=2,
        max_input_tokens=256,
    )
    _tokenizer, loaded_model, _torch, _stopping = provider._load_model()
    assert "default" in loaded_model.peft_config
    agent = Agent(
        AgentDefinition(
            name="local-adapter-acceptance",
            model={"provider": "transformers", "model": str(base_model_path)},
            policies={"max_steps": 1},
        ),
        model=provider,
    )
    try:
        result = await agent.arun("What is two plus two?")
    finally:
        await agent.aclose()
    assert any(event.kind == "model_response" for event in result.trace.events)
