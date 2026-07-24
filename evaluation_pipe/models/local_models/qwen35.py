"""Qwen3.5 natively-multimodal wrappers for shape-bias evaluation."""

from __future__ import annotations

import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3_5ForConditionalGeneration,
)

from evaluation_pipe.eval_core import (
    LOCAL_VLM_SYSTEM_PROMPT,
    build_transformers_vision_user_content,
)

from ..base import BaseVLM, ModelResponse
from .. import register_model


class _Qwen35Base(BaseVLM):
    """Shared loading/inference logic for Qwen3.5 vision-language models."""

    _default_model_id: str  # set by subclasses
    _system_prompt = LOCAL_VLM_SYSTEM_PROMPT
    # Override with "auto" on subclasses too large for one GPU so accelerate
    # shards the weights across all visible GPUs (bf16, no quantization).
    _device_map: str | None = None
    # Set True on subclasses that must be loaded in 4-bit (nf4) to fit one GPU.
    # The Gated-DeltaNet hybrid goes numerically unstable when sharded across
    # GPUs in bf16, so large variants run quantized on a single GPU instead.
    _quantization_4bit: bool = False

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        model_id = model_id or self._default_model_id
        self._model_id = model_id
        self._device = device
        device_map = self._device_map or device
        quantization_config = None
        if self._quantization_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            quantization_config=quantization_config,
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
        choice_texts: tuple[str, str] | None = None,
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
                return self._model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, output_scores=True)

        output_ids, elapsed = self._timed_generate(_run)

        # Assembling the generated response
        generated_token_ids = output_ids.sequences[:, input_len:]
        raw_text = self._processor.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
        num_tokens = generated_token_ids.shape[1]

        choice_logits = None
        choice_probs = None
        # Storing logits
        if choice_texts is not None:
            logits = output_ids.scores[0]
            
            # Token ids for the given text options
            id_1 = self._processor.tokenizer.encode(choice_texts[0], add_special_tokens=False)[0]
            id_2 = self._processor.tokenizer.encode(choice_texts[1], add_special_tokens=False)[0]

            logit_1 = logits[0, id_1]
            logit_2 = logits[0, id_2]

            probs = torch.softmax(logits, dim=-1)

            p_1 = probs[0, id_1]
            p_2 = probs[0, id_2]

            choice_logits = (logit_1.item(), logit_2.item())
            choice_probs = (p_1.item(), p_2.item())

        return ModelResponse(
            raw_text=raw_text.strip(),
            generation_time_s=elapsed,
            model_name=self.name,
            num_tokens_generated=num_tokens,
            choice_logits = choice_logits,
            choice_probs = choice_probs
        )
    
    def score_choices(
        self,
        images: list[Image.Image],
        prompt: str,
        choice_texts: tuple[str, str] = ("1", "2"),
        top_k: int = 0
    ) -> dict:
        """Return next-token probabilities/logits for two one-token choices."""
        content = build_transformers_vision_user_content(images, prompt)
        messages: list[dict] = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {"role": "user", "content": content},
        ]

        # Same two-step path as generate (enable_thinking=False via tokenizer).
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

        choice_ids: list[int] = []
        for ch in choice_texts:
            toks = self._processor.tokenizer.encode(ch, add_special_tokens=False)
            if len(toks) != 1:
                raise ValueError(f"Choice {ch!r} must map to one token; got ids={toks}")
            choice_ids.append(toks[0])

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._model(**inputs)

            next_logits = out.logits[:, -1, :]
            probs = torch.softmax(next_logits, dim=-1)
            topk = None
            if top_k > 0:
                topk = torch.topk(probs, top_k)

        elapsed = time.perf_counter() - t0

        next_logits = out.logits[:, -1, :].float()
        choice_logits = next_logits[0, choice_ids]

        all_probs = torch.softmax(next_logits, dim=-1)
        probs_absolute = all_probs[0, choice_ids]

        return {
            "choice_texts": list(choice_texts),
            "choice_token_ids": choice_ids,
            "choice_logits": [float(choice_logits[0].item()), float(choice_logits[1].item())],
            "choice_probs_absolute": [float(probs_absolute[0].item()), float(probs_absolute[1].item())],
            "generation_time_s": elapsed,
            "model_name": self.name,
            "topk": topk
        }

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


@register_model("qwen3.5-9b")
class Qwen35_9B(_Qwen35Base):
    """Qwen3.5-9B wrapper (next dense step above 4B in the passing family)."""

    _default_model_id = "Qwen/Qwen3.5-9B"


@register_model("qwen3.5-27b")
class Qwen35_27B(_Qwen35Base):
    """Qwen3.5-27B wrapper.

    Loaded in 4-bit (nf4) on a single GPU (~16 GB): bf16 needs ~55 GB, which
    only fits by sharding across GPUs, and sharding this Gated-DeltaNet hybrid
    is numerically unstable (nan logits). Results carry a quantization caveat
    relative to the bf16 rungs (4B / 9B).
    """

    _default_model_id = "Qwen/Qwen3.5-27B"
    _device_map = "auto"
    _quantization_4bit = True
