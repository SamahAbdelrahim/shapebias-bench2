"""Base class and response dataclass for Vision-Language Models."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class ModelResponse:
    """Container for a VLM generation result."""

    raw_text: str
    generation_time_s: float
    model_name: str
    num_tokens_generated: int


class BaseVLM(ABC):
    """Abstract base class for all VLM wrappers.

    Every concrete wrapper must implement __init__, name, and generate.
    The generate method accepts a **list** of images to support the 2AFC
    match-to-sample paradigm (reference + two comparison images).
    """

    @abstractmethod
    def __init__(self, model_id: str, device: str = "cuda", **kwargs) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical model name used in logs and results."""
        ...

    @abstractmethod
    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Run inference on *images* conditioned on *prompt*.

        Parameters
        ----------
        images : list[Image.Image]
            For the 2AFC task this will be [reference, image_a, image_b].
        prompt : str
            The text prompt (e.g. asking which image matches the reference).
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature (0.0 = greedy).
        """
        ...

    def unload(self) -> None:
        """Release GPU memory. Override in subclasses for cleanup."""
        pass

    # --- helpers available to all subclasses ---

    @staticmethod
    def _timed_generate(generate_fn):
        """Call *generate_fn* and return (output, elapsed_seconds)."""
        start = time.perf_counter()
        output = generate_fn()
        elapsed = time.perf_counter() - start
        return output, elapsed
