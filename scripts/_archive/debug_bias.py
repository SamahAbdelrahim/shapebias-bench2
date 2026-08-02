#!/usr/bin/env python3
"""Quick diagnostic: is SmolVLM's always-B behavior a text bias or positional bias?

Runs 2 trials on a single stimulus:
  1. Normal:  labels = A (first comparison), B (second comparison)
  2. Swapped: labels = B (first comparison), A (second comparison)

If the model says "B" in both cases → text bias (prefers letter "B")
If the model says "B" then "A"     → positional bias (prefers second image)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from evaluation_pipe.eval_core import load_stimuli, PROMPT_TEMPLATE

# Load one stimulus
stimuli = load_stimuli(num_stimuli=1)
stim = stimuli[0]
ref, shape, texture = stim["reference"], stim["shape_match"], stim["texture_match"]
word = "shiple"
prompt = PROMPT_TEMPLATE.format(word=word)

print(f"Prompt: {prompt}")
print(f"Stimulus: {stim['stim_id']}")
print()

# Load SmolVLM
from evaluation_pipe.models import create_model
model = create_model("smolvlm")

def run_with_content(content_list, label):
    """Run inference with a custom content list."""
    from transformers import AutoProcessor
    messages = [{"role": "user", "content": content_list}]
    inputs = model._processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model._device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model._model.generate(**inputs, max_new_tokens=128, do_sample=False)
    new_ids = output_ids[:, input_len:]
    raw_text = model._processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
    print(f"  {label}: raw_text={raw_text!r}")
    return raw_text


# --- Test 1: Normal order (A=first, B=second) ---
print("Test 1: Normal labels (A=shape, B=texture)")
content_normal = [
    {"type": "text", "text": "Reference image:"},
    {"type": "image", "image": ref},
    {"type": "text", "text": "Image A:"},
    {"type": "image", "image": shape},
    {"type": "text", "text": "Image B:"},
    {"type": "image", "image": texture},
    {"type": "text", "text": prompt},
]
r1 = run_with_content(content_normal, "Normal A/B")

# --- Test 2: Swapped labels (B=first, A=second) ---
print("Test 2: Swapped labels (B=shape, A=texture) — same image order!")
content_swapped = [
    {"type": "text", "text": "Reference image:"},
    {"type": "image", "image": ref},
    {"type": "text", "text": "Image B:"},
    {"type": "image", "image": shape},
    {"type": "text", "text": "Image A:"},
    {"type": "image", "image": texture},
    {"type": "text", "text": prompt},
]
r2 = run_with_content(content_swapped, "Swapped B/A")

# --- Test 3: No labels at all, ask to describe images ---
print("\nTest 3: Ask model to describe what it sees (verify images are processed)")
content_describe = [
    {"type": "image", "image": ref},
    {"type": "image", "image": shape},
    {"type": "image", "image": texture},
    {"type": "text", "text": "Briefly describe each of the three images you see."},
]
run_with_content(content_describe, "Describe")

# --- Test 4: Use 1/2 instead of A/B ---
print("\nTest 4: Use '1' and '2' instead of 'A' and 'B'")
prompt_12 = (
    f"The first image is a {word}. "
    f"Which of the following two images (1 or 2) is also a {word}? "
    "Answer with just '1' or '2'."
)
content_12 = [
    {"type": "text", "text": "Reference image:"},
    {"type": "image", "image": ref},
    {"type": "text", "text": "Image 1:"},
    {"type": "image", "image": shape},
    {"type": "text", "text": "Image 2:"},
    {"type": "image", "image": texture},
    {"type": "text", "text": prompt_12},
]
run_with_content(content_12, "Numeric 1/2")

print("\n--- Diagnosis ---")
if r1 == "B" and r2 == "B":
    print("TEXT BIAS: model always outputs 'B' regardless of label position.")
    print("Consider using numeric labels (1/2) or counterbalancing A/B assignment.")
elif r1 == "B" and r2 == "A":
    print("POSITIONAL BIAS: model always picks the second/last comparison image.")
    print("Consider counterbalancing image order (already done via ordering param).")
else:
    print(f"Mixed results: normal={r1!r}, swapped={r2!r} — further investigation needed.")

model.unload()
