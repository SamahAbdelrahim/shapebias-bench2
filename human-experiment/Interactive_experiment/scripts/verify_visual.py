#!/usr/bin/env python3
"""Verify that model responses are driven by visual content, not artifacts.

Test 1 — DETERMINISM: Same input twice → same output (no hidden randomness).
Test 2 — IMAGE SENSITIVITY: Swap image positions → answer should follow the
         correct image. If model picks "1" when shape is first, it should pick
         "2" when shape is second (i.e. it tracks the shape image, not a fixed
         position).
Test 3 — WORD SENSITIVITY: Same images, different word → can produce different
         answers (rules out the model ignoring the text).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from evaluation_pipe.eval_core import load_stimuli, PROMPT_TEMPLATE
from evaluation_pipe.models import create_model, list_models


def run_once(model, ref, img1, img2, prompt):
    content = [
        {"type": "text", "text": "Reference image:"},
        {"type": "image", "image": ref},
        {"type": "text", "text": "Image 1:"},
        {"type": "image", "image": img1},
        {"type": "text", "text": "Image 2:"},
        {"type": "image", "image": img2},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    inputs = model._processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model._device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model._model.generate(**inputs, max_new_tokens=128, do_sample=False)
    return model._processor.batch_decode(out[:, input_len:], skip_special_tokens=True)[0].strip()


parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default="smolvlm",
                    help=f"Model to test. Available: {list_models()}")
args = parser.parse_args()

print(f"Model: {args.model}\n")
model = create_model(args.model)
stimuli = load_stimuli(num_stimuli=10)
words = ["shiple", "clapher", "plailass", "procation", "adinefults"]

# =========================================================================
# Test 1: DETERMINISM — run identical input twice
# =========================================================================
print("=" * 65)
print("TEST 1: DETERMINISM (same input twice → same output?)")
print("=" * 65)
determinism_pass = 0
determinism_total = 0
for stim in stimuli[:5]:
    ref, shape, texture = stim["reference"], stim["shape_match"], stim["texture_match"]
    prompt = PROMPT_TEMPLATE.format(word="shiple")
    r1 = run_once(model, ref, shape, texture, prompt)
    r2 = run_once(model, ref, shape, texture, prompt)
    match = r1 == r2
    determinism_pass += match
    determinism_total += 1
    status = "PASS" if match else "FAIL"
    print(f"  stim={stim['stim_id']:>3s}  run1={r1!r}  run2={r2!r}  [{status}]")

print(f"\n  Result: {determinism_pass}/{determinism_total} deterministic")

# =========================================================================
# Test 2: IMAGE SENSITIVITY — swap positions, answer should follow image
# =========================================================================
print()
print("=" * 65)
print("TEST 2: IMAGE SENSITIVITY (swap positions → answer follows image?)")
print("=" * 65)
print("  If model picks shape, it should say '1' when shape is first,")
print("  '2' when shape is second.")
print()

tracks_image = 0
tracks_position = 0
inconsistent = 0
total = 0

for stim in stimuli:
    ref, shape, texture = stim["reference"], stim["shape_match"], stim["texture_match"]
    prompt = PROMPT_TEMPLATE.format(word="shiple")

    # shape_first: Image 1 = shape, Image 2 = texture
    r_sf = run_once(model, ref, shape, texture, prompt)
    # texture_first: Image 1 = texture, Image 2 = shape
    r_tf = run_once(model, ref, texture, shape, prompt)

    total += 1

    if r_sf == "1" and r_tf == "2":
        # Picks shape in both orderings → tracks the shape image
        verdict = "TRACKS IMAGE (shape)"
        tracks_image += 1
    elif r_sf == "2" and r_tf == "1":
        # Picks texture in both orderings → tracks the texture image
        verdict = "TRACKS IMAGE (texture)"
        tracks_image += 1
    elif r_sf == r_tf:
        # Same answer regardless of swap → position/label bias
        verdict = f"POSITION BIAS (always {r_sf})"
        tracks_position += 1
    else:
        verdict = "INCONSISTENT"
        inconsistent += 1

    print(f"  stim={stim['stim_id']:>3s}  shape_first={r_sf!r}  texture_first={r_tf!r}  → {verdict}")

print(f"\n  Result: {tracks_image}/{total} track image, "
      f"{tracks_position}/{total} position bias, "
      f"{inconsistent}/{total} inconsistent")

# =========================================================================
# Test 3: WORD SENSITIVITY — same images, different words
# =========================================================================
print()
print("=" * 65)
print("TEST 3: WORD SENSITIVITY (different words → any answer variation?)")
print("=" * 65)

stim = stimuli[0]
ref, shape, texture = stim["reference"], stim["shape_match"], stim["texture_match"]
answers = []
for w in words:
    prompt = PROMPT_TEMPLATE.format(word=w)
    r = run_once(model, ref, shape, texture, prompt)
    answers.append(r)
    print(f"  word={w:>12s}  -> {r!r}")

unique = len(set(answers))
print(f"\n  Result: {unique} unique answer(s) across {len(words)} words")
if unique > 1:
    print("  Words DO influence the answer.")
else:
    print("  Words do NOT influence the answer (same response for all words).")

# =========================================================================
# Summary
# =========================================================================
print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  Determinism:      {determinism_pass}/{determinism_total}")
print(f"  Tracks image:     {tracks_image}/{total}")
print(f"  Position bias:    {tracks_position}/{total}")
print(f"  Word sensitivity: {unique}/{len(words)} unique answers")

if determinism_pass == determinism_total and tracks_image > tracks_position:
    print("\n  CONCLUSION: Responses are deterministic and driven by visual content.")
elif determinism_pass < determinism_total:
    print("\n  CONCLUSION: Non-deterministic outputs — possible hidden randomness.")
elif tracks_position >= tracks_image:
    print("\n  CONCLUSION: Mostly position bias — model is not reliably using images.")

model.unload()
