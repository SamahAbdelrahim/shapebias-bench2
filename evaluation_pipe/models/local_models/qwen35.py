"""Qwen3.5 natively-multimodal wrappers for shape-bias evaluation."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from evaluation_pipe.eval_core import (
    QWEN35_VLM_SYSTEM_PROMPT,
    build_transformers_vision_user_content,
)

from ..base import BaseVLM, ModelResponse
from .. import register_model


class _Qwen35Base(BaseVLM):
    """Shared loading/inference logic for Qwen3.5 vision-language models."""

    _default_model_id: str  # set by subclasses
    _system_prompt = QWEN35_VLM_SYSTEM_PROMPT

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        model_id = model_id or self._default_model_id
        self._model_id = model_id
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            **kwargs,
        )
        self._model.eval()

    @property
    def name(self) -> str:
        return self._model_id

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelResponse:
        content = build_transformers_vision_user_content(images, prompt)
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {"role": "user", "content": content},
        ]

        # Two-step tokenization: the processor's apply_chat_template does not
        # forward enable_thinking to the Jinja2 template, so we call the
        # tokenizer first (to get the text with thinking disabled) and then
        # pass the rendered text + raw images to the processor for encoding.
        text = self._processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        inputs = self._processor(
            text=[text],
            images=list(images),
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = dict(max_new_tokens=max_new_tokens)
        if temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature

        def _run():
            with torch.inference_mode():
                return self._model.generate(**inputs, **gen_kwargs)

        output_ids, elapsed = self._timed_generate(_run)

        new_ids = output_ids[:, input_len:]
        num_tokens = new_ids.shape[1]
        raw_text = self._processor.batch_decode(new_ids, skip_special_tokens=True)[0]

        return ModelResponse(
            raw_text=raw_text.strip(),
            generation_time_s=elapsed,
            model_name=self.name,
            num_tokens_generated=num_tokens,
        )

    def unload(self) -> None:
        del self._model
        del self._processor
        torch.cuda.empty_cache()


@register_model("qwen3.5-0.8b")
class Qwen35_08B(_Qwen35Base):
    """Qwen3.5-0.8B wrapper."""

    _default_model_id = "Qwen/Qwen3.5-0.8B"


@register_model("qwen3.5-4b")
class Qwen35_4B(_Qwen35Base):
    """Qwen3.5-4B wrapper."""

    _default_model_id = "Qwen/Qwen3.5-4B"

@register_model("qwen3.5-1.7b")
class Qwen35_17B(_Qwen35Base):
    """Qwen3.5-1.7B-Instruct wrapper."""

    _default_model_id = "Qwen/Qwen3.5-1.7B-Instruct"
