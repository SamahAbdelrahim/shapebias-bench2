"""SmolVLM wrapper for shape-bias evaluation."""

from __future__ import annotations

import time

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
    SmolVLMProcessor,
)

from evaluation_pipe.eval_core import (
    LOCAL_VLM_SYSTEM_PROMPT,
    build_transformers_vision_user_content,
)

from ..base import BaseVLM, ModelResponse
from .. import register_model

_DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
_DEFAULT_MODEL_ID_256M = "HuggingFaceTB/SmolVLM-256M-Instruct"


@register_model("smolvlm")
class SmolVLM(BaseVLM):
    """Wrapper for SmolVLM2-2.2B-Instruct (multi-image native)."""

    _system_prompt = LOCAL_VLM_SYSTEM_PROMPT

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self._model_id = model_id
        self._device = device
        load_dtype = torch.bfloat16 if device.startswith(("cuda", "mps")) else torch.float32

        def _load_model(dtype: torch.dtype):
            # Newer transformers releases can fail to auto-detect the image processor
            # for SmolVLM2. Prefer explicit SmolVLM classes, then fall back to Auto*.
            try:
                processor = SmolVLMProcessor.from_pretrained(model_id)
                model = SmolVLMForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    **kwargs,
                )
            except Exception:
                processor = AutoProcessor.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                    **kwargs,
                )
            return processor, model

        self._processor, self._model = _load_model(load_dtype)
        self._model = self._model.to(device)
        self._device = device
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
        choice_texts: tuple[str, str] | None = None
    ) -> ModelResponse:
        content = build_transformers_vision_user_content(images, prompt)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {"role": "user", "content": content},
        ]

        input_dtype = torch.bfloat16 if self._device.startswith(("cuda", "mps")) else torch.float32

        # Two-step encode avoids kwargs warnings emitted by SmolVLMProcessor
        # when tokenization kwargs are forwarded through apply_chat_template.
        text = self._processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=[text],
            images=list(images),
            return_tensors="pt",
        ).to(self._device, dtype=input_dtype)

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
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {"role": "user", "content": content},
        ]
        input_dtype = torch.bfloat16 if self._device.startswith(("cuda", "mps")) else torch.float32

        text = self._processor.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self._processor(
            text=[text],
            images=list(images),
            return_tensors="pt",
        ).to(self._device, dtype=input_dtype)

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


@register_model("smolvlm-256m")
class SmolVLM256M(SmolVLM):
    """Wrapper for SmolVLM-256M-Instruct (smaller baseline)."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID_256M,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        super().__init__(model_id=model_id, device=device, **kwargs)
