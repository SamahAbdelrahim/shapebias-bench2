# Shape bias model behavior discussion
_Exported on 4/10/2026 at 11:02:42 PDT from Cursor (3.0.13)_

---

**User**

@/Users/samahabdelrahim/git-repos/shapebias-bench-2/ Samah’s shape bias benchmarking project is working well but running into a few problems with making models “behave” - follow prompt and do sensical things with a group of images. I wonder if anyone would have a few minutes to chat with her about some of these issues. bigger Qwen models are behaving ok but InternVL and SmolVLM seem to show a very strong side bias… we’re wondering if some prompt optimization would at least help them pass her control tasks so she can even assess whether they have a shape bias. Thanks! (edited) I've been experimenting with shapebias on smolvlm2 (2.2B like your code). It is unbelievably stubborn. I've switched the order of the prompt image numbers, split the prompts into two, and a couple other things. I'm trying some more options, but so far it seems like "it doesn't have a clue?" "Just in case" I tried MAX_TOKENS from 1 to 512 for smolvlm, and I tried the 256M version of smolvlm. All with no joy. @Samah Some decent progress. If we carefully "curate" the way smolvlm2 answers we can get useful signal. Below is the summary. Code is in the fork I made from your repo:

TL;DR -- Sampling the precise next logit to see if it is 1 or 2 appears to make a huge difference compared to asking the model for "an answer."

• In the old setup, we asked the model to generate text (like "1" or "2").
That generation is noisy and prompt-sensitive, so the model often defaulted to a habit (e.g., always first option), not true visual comparison.
• In logit_forced_12, we don’t trust the generated sentence.
Instead, we look at the model’s immediate preference scores for token 1 vs token 2 at the answer position (its logits/probabilities), then pick whichever is higher.
So it’s like:

old: “say the answer out loud” (can ramble or follow a bad habit)
new: “peek at your internal vote: 1 or 2?”

Why --swap-correct matters
Even with logits, the model can still have a positional prior (e.g., likes the first slot).
So we run each trial twice:
normal order: [ref, A, B] -> get preference for 1 vs 2

swapped order: [ref, B, A] -> get another preference
Then we remap the swapped result back to original semantics and average.
That cancels “slot bias” because A and B each appear once in slot 1 and once in slot 2.

Concrete meaning of your result
Without swap-correct, strong order bias remained.

With swap-correct, order effects nearly vanished (shape_first and texture_first became similar).
That suggests the model may still contain some real signal, but it was previously hidden by positional bias.
So this is a measurement fix, not a “model became better” fix. It makes your estimate more trustworthy. Looking for the logits on the right answer is a second method that alvin tried in dev bench! I think that method could be very useful more generally for reducing “task demands “on our models…
david_cardinal  [5:22 PM]
After a full run of smol using the logits code, it actually has a bit of a texture bias! Overall it is still below chance:

ide/order bias is effectively removed:second_pick_rate = 50.0%
shape_rate_shape_first = 40.4%
shape_rate_texture_first = 40.4%
order gap = 0.0%

De-biased outcome:smolvlm2_adjusted_score = 40.4%
adjusted_shape_bias_index = -0.096


shapebias repo uses a very different framework for model registration and running than levante-bench. So I've created a branch of levante-bench (integrate-shapebias) that I hope pulls over all the "science" portions of shapebias, but lets them run inside our levante-bench setup. If it works it should make life simpler for anyone wanting to work on both. @/Users/samahabdelrahim/git-repos/levante-bench/ 

i need to understand this different framework in levante-bench shape bias version and the original shapebias-bench-2. 


---

**Cursor**


Exploring both repositories to compare how models are registered and run in the original shapebias-bench-2 versus levante-bench’s shape-bias integration.




Here is a concise comparison of the two frameworks and how shape bias fits in each.

## 1. Big picture

| Aspect | **shapebias-bench-2** | **levante-bench (shapebias integration)** |
|--------|------------------------|-------------------------------------------|
| **Entry point** | `scripts/run_evaluation.py` (argparse, self-contained) | `levante-bench run-eval` + OmegaConf **experiment** configs |
| **What drives a run** | CLI flags + constants in the script | `configs/experiment.yaml` lists **models** and **tasks**; merges **model YAML** + **task YAML** + overrides |
| **Shape bias “science”** | Inline in the script + `evaluation_pipe` helpers | `tasks/shapebias.py` (trials) + `evaluation/shapebias.py` (prompts, `logit_forced_12`, swap) + runner hook |

The integration branch’s goal is the same experiments (stimuli, words, decision modes) but **inside levante’s single runner**, shared caching, `results/<version>/<model>-<size>/`, and the rest of the benchmark tasks.

---

## 2. Model registration

**Both** use a small decorator registry (`name → class`), but names and wiring differ.

**shapebias-bench-2** — `register_model` + `create_model`:

```13:38:/Users/samahabdelrahim/git-repos/shapebias-bench-2/evaluation_pipe/models/__init__.py
def register_model(name: str):
    """Class decorator that registers a VLM wrapper under *name*."""

    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def create_model(name: str, **kwargs) -> BaseVLM:
    ...
    return MODEL_REGISTRY[name](**kwargs)
```

Local wrappers live under `evaluation_pipe/models/local_models/` and are pulled in via `local_models/__init__.py` so decorators run.

**levante-bench** — `register` + `get_model_class`, with classes **explicitly imported** in `models/__init__.py`:

```10:18:/Users/samahabdelrahim/git-repos/levante-bench/src/levante_bench/models/__init__.py
# Import model modules so @register() runs
from levante_bench.models import vlm  # noqa: F401
from levante_bench.models import smolvlm2  # noqa: F401
from levante_bench.models import qwen35  # noqa: F401
from levante_bench.models import internvl35  # noqa: F401
```

Model **hyperparameters and HF ids** live in **`configs/models/<model>.yaml`** and are merged at run time (`load_model_config`), not only in the wrapper’s constructor.

---

## 3. Base interface (this is the important behavioral difference)

**shapebias-bench-2** — `BaseVLM`: PIL images in, `ModelResponse` out.

```22:46:/Users/samahabdelrahim/git-repos/shapebias-bench-2/evaluation_pipe/models/base.py
class BaseVLM(ABC):
    ...
    @abstractmethod
    def generate(
        self,
        images: list[Image.Image],
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
    ) -> ModelResponse:
```

**levante-bench** — `VLMModel`: trial-oriented API with **paths**, optional **JSON** instructions for most tasks, and **`score_choices`** for forced logits (used by `logit_forced_12`).

```68:76:/Users/samahabdelrahim/git-repos/levante-bench/src/levante_bench/models/base.py
    def score_choices(
        self,
        prompt_text: str,
        image_paths: list[str] | None = None,
        choice_texts: tuple[str, str] = ("1", "2"),
    ) -> dict:
        """Optional: return next-token logits/probs for constrained choices."""
        raise NotImplementedError("score_choices is not implemented for this model.")
```

So the “same” SmolVLM-style wrapper is not a drop-in: it must implement levante’s `load` / `generate` / optional `score_choices` and match how trials are built.

---

## 4. How a run is orchestrated

**shapebias-bench-2**

- **Local**: `create_model(model_key, device=...)` → loop stimuli × words → `run_local` / `run_trial`.
- **Remote**: a separate **`REMOTE_MODELS`** dict in `run_evaluation.py` maps short names to provider + `model_id`; that path does not use `MODEL_REGISTRY` the same way.

**levante-bench**

- One **`run_eval`**: for each model in the experiment config, load YAML, **`get_model_class` → construct → `load()`**, then for each **task** in the config, build a **registered dataset** and evaluate.

Shape bias is **not** handled only by `evaluate_trial` on the generic path. The runner **special-cases** the shapebias task:

```264:267:/Users/samahabdelrahim/git-repos/levante-bench/src/levante_bench/evaluation/runner.py
                if task_id == "shapebias":
                    result = evaluate_shapebias_trial(model, trial)
                else:
                    result = model.evaluate_trial(trial)
```

So shapebias shares the same model object and caching as TROG, egma-math, etc., but trial evaluation goes through `evaluation/shapebias.py` instead of the default JSON letter-parsing pipeline.

---

## 5. Configuration

- **shapebias-bench-2**: stimulus dir, device, seeds, etc. mostly via **CLI + script constants**; prompt template in-script (e.g. A/B wording in the snippet you saw).
- **levante-bench**: **`configs/tasks/shapebias.yaml`** for defaults (`ordering`, `decision_mode`, `swap_correct`, …), plus CLI flags like `--shapebias-stim-root` that become **task overrides** (see `cli.py` around the shapebias arguments). Model-level settings (`use_json_format: false` for SmolVLM, sizes, etc.) sit in **`configs/models/`**.

---

## 6. What “integrate-shapebias” changes in practice

- **One codebase** for model adapters, results layout, and multi-task runs.
- Shape bias **logic** (prompts, `logit_forced_12`, swap correction) is centralized in **`levante_bench.evaluation.shapebias`** + **`ShapeBiasDataset`**, not in a standalone script.
- You still need to align **adapter capabilities** (especially `score_choices` for logits) with what `evaluate_shapebias_trial` expects—that is the main “porting” work versus copying `run_evaluation.py` wholesale.

If you want to go deeper on one axis (e.g. exact parity of prompt strings A/B vs 1/2, or CSV columns vs `shapebias_detailed.csv`), say which repo file you care about and we can line them up line by line.

---

**User**

explain this to me: Completely agree. What I was mostly thinking about was that if these efforts will be connected having a single way to register and run models/experiments would be handy. I mostly created the branch for myself so that I could run shapebias using the variety of models we have in -bench without setting them up all over again. No clue if that is actually important longer term. Mike, what are the consequences of merging both repos? Does this mean i will work on the levante-bench repo but would still be able to play around on the shape bias work on my own without touching anything related to Levante?
Mike Frank  [9:34 AM]
Yes that’s right, but ideally you would be able to make use of some of the model harnesses and parsers that the group is developing.

---

**Cursor**


Here is a plain-language reading of that thread.

## What “single way to register and run” means

Right now you effectively have **two setups**:

- **shapebias-bench-2**: its own script, its own `create_model` / `BaseVLM` stack, and (for APIs) its own remote registry baked into `run_evaluation.py`.
- **levante-bench**: a **shared** pattern—YAML experiment + task configs, `levante-bench run-eval`, one `VLMModel` interface, one results layout, shared adapters for SmolVLM, Qwen, InternVL, etc.

“Connected efforts” and “single way” means: **one place** where a model is defined, loaded, and invoked so you are not re-implementing or re-wiring the same HF models twice. The **integrate-shapebias** style branch is exactly that: shape bias **experiments** ride on levante’s runner so you pick `--model smolvlm2` (or whatever is already in `configs/models/`) instead of maintaining a parallel harness in the shapebias repo.

Whether that matters long term depends on whether the group keeps investing in **one** benchmark codebase. If levante-bench is where new parsers, logits APIs, and model fixes land, duplicating them in shapebias-bench-2 is ongoing friction.

## What the branch was for (your note)

You wanted to **run shape bias with the same model zoo** levante already has, **without** registering every model again in the old repo. That is a practical, “don’t repeat yourself” motivation—not necessarily a mandate to delete shapebias-bench-2.

## Mike’s merge question (two parts)

**If everything lives in levante-bench long term**

- Day-to-day work on **models + shape bias in that integration** would happen **in levante-bench** (configs, `evaluation/shapebias.py`, adapters).
- That does **not** automatically mean you can never touch “pure” shape bias again; it means the **supported** path for *combined* model + shape bias work is one repo.

**“Still play with shape bias on my own without touching Levante”**

- **Yes, in principle**, you can keep **shapebias-bench-2** (or a fork) as a **small, focused** repo for stimuli, analysis, or quick scripts that do not depend on levante.
- **Caveat**: if the group **merges** and only maintains **one** codebase, then “shape bias only” changes might either (a) move to levante under `tasks/shapebias` + data paths, or (b) live in a separate repo that **duplicates** or **vendors** pieces—unless you explicitly keep two maintained surfaces (costly).

So merge **consequences** are mostly **social and maintenance**: one source of truth for harnesses, fewer places to fix bugs, but you need discipline about what belongs in “core levante” vs optional/experimental.

## Mike Frank’s reply (“harnesses and parsers”)

He is aligning with **not** forcing everything into one monolith for its own sake, but saying the **value** of connecting work is **reuse**: logits / `score_choices`-style evaluation, response parsing, JSON handling, caching, CLI patterns—the **machinery** the group improves in levante-bench—so shape bias (and you) **benefit** from that shared layer instead of reinventing it in isolation.

**Bottom line:** Merging (or standardizing on levante for runs) means you’d **usually** work in levante-bench for integrated model + shape bias runs; you can still keep a **separate** shapebias playground, but anything “official” or shared with the group would naturally gravitate to where the harnesses live—unless the group explicitly maintains two parallel systems.

