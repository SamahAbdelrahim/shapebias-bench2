"""Adapter that runs local inference through levante_bench runtime APIs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from PIL import Image

from ..base import BaseVLM, ModelResponse
from .. import register_model


@register_model("levante-runtime")
class LevanteRuntimeVLM(BaseVLM):
    """ShapeBias wrapper that delegates generation to levante_bench models."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str = "cuda",
        **kwargs,
    ) -> None:
        try:
            from levante_bench.runtime import load_model, run_trials
        except ImportError as exc:
            raise ImportError(
                "levante_bench is not installed. Install from sibling repo with "
                "`pip install -e /home/david/levante/levante-bench` (or your local path)."
            ) from exc

        self._run_trials = run_trials
        configured_model = model_id or os.environ.get("LEVANTE_MODEL_NAME", "qwen35")
        model_config_path = kwargs.get("model_config_path") or os.environ.get(
            "LEVANTE_MODEL_CONFIG_PATH"
        )
        configs_root = kwargs.get("configs_root") or os.environ.get("LEVANTE_CONFIGS_ROOT")
        self._levante_model_name = configured_model
        self._model = load_model(
            model_name=configured_model,
            model_config_path=model_config_path,
            configs_root=configs_root,
            device=device,
            auto_load=True,
        )

    @property
    def name(self) -> str:
        return f"levante::{self._levante_model_name}"

    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelResponse:
        # levante runtime currently uses deterministic generation in wrappers;
        # preserve ShapeBias signature while ignoring temperature here.
        del temperature

        with tempfile.TemporaryDirectory(prefix="shapebias-levante-") as tmp_dir:
            image_paths: list[str] = []
            for i, image in enumerate(images):
                img_path = Path(tmp_dir) / f"image_{i}.png"
                image.save(img_path, format="PNG")
                image_paths.append(str(img_path))

            trial = {
                "trial_id": "shapebias-runtime-trial",
                "item_uid": "shapebias-runtime-trial",
                "prompt": prompt,
                "option_labels": ["1", "2"],
                "correct_label": "1",
                "context_image_paths": image_paths,
                "option_image_paths": [],
                "answer_format": "label",
                "max_new_tokens": max_new_tokens,
            }
            result = self._run_trials(
                model=self._model,
                trials=[trial],
                max_new_tokens=max_new_tokens,
                task_id="shape-bias",
            )[0]

        return ModelResponse(
            raw_text=str(result.get("generated_text", "")),
            generation_time_s=0.0,
            model_name=self.name,
            num_tokens_generated=0,
        )

    def unload(self) -> None:
        # levante wrappers do not currently expose explicit unload hooks.
        self._model = None
