"""TinyLLaVA wrapper for shape-bias evaluation.

DEPRECATED: TinyLLaVA's custom HuggingFace model code is incompatible with
the current transformers version (SigLIP architecture mismatch causes CUDA
errors at inference time). This module is kept for reference but is not
imported by default.

TinyLLaVA only supports single-image input. For the 2AFC paradigm we compose
all images into a horizontal collage before sending them to the model.
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from ..base import BaseVLM, ModelResponse
from .. import register_model

_DEFAULT_MODEL_ID = "tinyllava/TinyLLaVA-Phi-2-SigLIP-3.1B"


@contextmanager
def _compat_tie_weights():
    """Temporarily patch ``PreTrainedModel.init_weights`` so that custom
    models whose ``tie_weights()`` doesn't accept the kwargs added by
    newer transformers (``recompute_mapping``, ``missing_keys``) still load.

    During ``init_weights`` we also permanently patch ``tie_weights`` on
    the *model's own class* so that the later call from
    ``_finalize_model_loading`` is safe too.
    """
    _orig_init_weights = PreTrainedModel.init_weights

    def _patched_init_weights(self):
        # Permanently patch tie_weights on this model's class to accept **kwargs
        model_cls = type(self)
        orig_tie = model_cls.tie_weights
        if not getattr(orig_tie, "_kwargs_safe", False):
            def _safe_tie(self, **kwargs):
                return orig_tie(self)
            _safe_tie._kwargs_safe = True
            model_cls.tie_weights = _safe_tie

        # Original init_weights logic
        if hasattr(self, "initialize_weights"):
            self.initialize_weights()
        self.tie_weights(recompute_mapping=False)

    PreTrainedModel.init_weights = _patched_init_weights
    try:
        yield
    finally:
        PreTrainedModel.init_weights = _orig_init_weights


def _make_collage(images: list[Image.Image], gap: int = 20) -> Image.Image:
    """Stitch *images* into a single horizontal strip with *gap* px spacing."""
    # Normalise heights so all images align
    target_h = max(img.height for img in images)
    resized = []
    for img in images:
        if img.height != target_h:
            scale = target_h / img.height
            new_w = int(img.width * scale)
            img = img.resize((new_w, target_h), Image.LANCZOS)
        resized.append(img)

    total_w = sum(img.width for img in resized) + gap * (len(resized) - 1)
    collage = Image.new("RGB", (total_w, target_h), (0, 0, 0))
    x = 0
    for img in resized:
        collage.paste(img, (x, 0))
        x += img.width + gap
    return collage


@register_model("tinyllava")
class TinyLLaVA(BaseVLM):
    """Wrapper for TinyLLaVA-Phi-2-SigLIP-3.1B (single-image, uses collage)."""

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        self._model_id = model_id
        self._device = device

        with _compat_tie_weights():
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="eager",
                **kwargs,
            ).to(device)
        self._model.eval()

        config = self._model.config
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=False,
            model_max_length=config.tokenizer_model_max_length,
            padding_side=config.tokenizer_padding_side,
        )

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
        # TinyLLaVA only supports a single image; compose into a collage.
        collage = _make_collage(images)

        # model.chat() expects a file path or URL, so write to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            collage.save(f, format="PNG")
            collage_path = f.name

        def _run():
            with torch.inference_mode():
                return self._model.chat(
                    prompt=prompt,
                    image=collage_path,
                    tokenizer=self._tokenizer,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

        (raw_text, gen_time), elapsed = self._timed_generate(_run)

        # Approximate token count from tokenizer
        num_tokens = len(self._tokenizer.encode(raw_text, add_special_tokens=False))

        return ModelResponse(
            raw_text=raw_text.strip(),
            generation_time_s=elapsed,
            model_name=self.name,
            num_tokens_generated=num_tokens,
        )

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        torch.cuda.empty_cache()
