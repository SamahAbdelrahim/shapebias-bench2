"""Data loading for 2AFC shape-bias evaluation trials."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class Trial:
    """A single 2AFC trial with reference + two comparison images."""

    trial_id: int
    mode: str
    order: str  # "shape_first" or "texture_first"
    reference_path: Path
    image_a_path: Path
    image_b_path: Path
    ground_truth: str  # "A" if shape_match is position A, else "B"

    def load_images(self) -> tuple[Image.Image, Image.Image, Image.Image]:
        """Load and return (reference, image_a, image_b) as RGB PIL images."""
        ref = Image.open(self.reference_path).convert("RGB")
        img_a = Image.open(self.image_a_path).convert("RGB")
        img_b = Image.open(self.image_b_path).convert("RGB")
        return ref, img_a, img_b


def load_trials(
    dataset_dir: str | Path,
    order: str = "shape_first",
    seed: int | None = None,
) -> list[Trial]:
    """Load all trials from a stimulus package directory.

    Discovers trials by scanning for numbered subdirectories containing
    reference.png, shape_match.png, and texture_match.png.

    Args:
        dataset_dir: Path to stimulus package (numbered folders with images).
        order: Image ordering strategy.
            - "shape_first": shape_match is always image A (ground_truth="A")
            - "texture_first": texture_match is always image A (ground_truth="B")
            - "random": each trial gets a random order (use seed for reproducibility)
        seed: Random seed, only used when order="random".

    Returns:
        List of Trial objects sorted by trial_id.
    """
    dataset_dir = Path(dataset_dir)

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    if order not in ("shape_first", "texture_first", "random"):
        raise ValueError(f"order must be 'shape_first', 'texture_first', or 'random', got '{order}'")

    rng = random.Random(seed) if order == "random" else None
    mode = dataset_dir.name

    # Discover trial folders: numbered subdirectories with the expected images
    trial_dirs = sorted(
        (d for d in dataset_dir.iterdir() if d.is_dir() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )

    if not trial_dirs:
        raise FileNotFoundError(f"No numbered trial folders found in {dataset_dir}")

    trials: list[Trial] = []
    for trial_dir in trial_dirs:
        stl_id = int(trial_dir.name)
        reference_path = trial_dir / "reference.png"
        shape_path = trial_dir / "shape_match.png"
        texture_path = trial_dir / "texture_match.png"

        if order == "random":
            trial_order = rng.choice(["shape_first", "texture_first"])
        else:
            trial_order = order

        if trial_order == "shape_first":
            image_a_path = shape_path
            image_b_path = texture_path
            ground_truth = "A"
        else:
            image_a_path = texture_path
            image_b_path = shape_path
            ground_truth = "B"

        trials.append(
            Trial(
                trial_id=stl_id,
                mode=mode,
                order=trial_order,
                reference_path=reference_path,
                image_a_path=image_a_path,
                image_b_path=image_b_path,
                ground_truth=ground_truth,
            )
        )

    return trials
