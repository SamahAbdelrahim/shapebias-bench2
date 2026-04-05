"""InternVL wrapper for shape-bias evaluation."""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation_pipe.eval_core import build_transformers_vision_user_content

from ..base import BaseVLM, ModelResponse
from .. import register_model

_DEFAULT_MODEL_ID = "OpenGVLab/InternVL3-1B-hf"


@register_model("internvl")
class InternVL(BaseVLM):
    """Wrapper for InternVL3 -hf variants (native transformers, multi-image)."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self._model_id = model_id
        self._device = device
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
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
        messages = [{"role": "user", "content": content}]

        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device, dtype=torch.bfloat16)

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
        raw_text = self._processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        num_tokens = new_ids.shape[1]

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
