#!/bin/bash
# LEACE erasure at the projector output: the causal test of whether the LM
# reads the encoder's mesh-identity subspace.
#
# Focal models (the cleanest gate-passing family) x five arms:
#   shape    erase the mesh-identity subspace (fold-aware, held-out meshes)
#   texture  erase the texture-identity subspace (complementary manipulation)
#   rand0-2  rank-matched random subspaces (three seeds)
#
# Only gate-passing prompt cells run (from results/data/full_grid_v1a_summary.csv,
# hardcoded below to keep the job self-contained). Outcome: drop in behavioral
# shape rate under shape-erasure vs both controls.
#
# Depends on the proj_lm token export (scripts/run_projlm_export.sbatch);
# eraser banks are fitted on CPU at job start if missing.
#
#   sbatch --dependency=afterany:<projlm_job> scripts/run_grid_erasure.sbatch
#
#SBATCH --job-name=sb-erase
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --array=0-9%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=oat-01
#SBATCH --output=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/erase_%A_%a.out
#SBATCH --error=/home/users/samahabd/shapebias-bench2/results/model.results/jobs/erase_%A_%a.err

set -euo pipefail
cd /home/users/samahabd/shapebias-bench2
source .venv/bin/activate
mkdir -p results/model.results/jobs
export PYTHONUNBUFFERED=1

export HF_HOME=/scratch/users/samahabd/hf_cache/huggingface
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GRID_PKG=stimuli_texture_grid_v1_scratch
STIM_SET=stimuli_A_auto_contrast
FIXED_WORD=shiple
EMB=results/probe.results/session_readout_power/embeddings_grid
BANKS=results/probe.results/session_readout_power/erasers

FOCAL=(qwen3.5-4b qwen3.5-9b)
ARMS=(shape texture rand0 rand1 rand2)

MODEL="${FOCAL[$((SLURM_ARRAY_TASK_ID / ${#ARMS[@]}))]}"
ARM="${ARMS[$((SLURM_ARRAY_TASK_ID % ${#ARMS[@]}))]}"

# Gate-passing generation cells per focal model (full_grid_v1a_summary.csv).
if [[ "$MODEL" == "qwen3.5-4b" ]]; then
  CONDITIONS=(no_word_similarity no_word_similarity_AB no_word_category
              no_word_category_AB noun_label noun_label_AB)
else
  CONDITIONS=(no_word_similarity no_word_category no_word_category_AB
              noun_label noun_label_AB)
fi

echo "host=$(hostname) date=$(date -Is) task=$SLURM_ARRAY_TASK_ID model=$MODEL arm=$ARM"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

TOKENS="$EMB/${MODEL}_proj_lm_tokens.npz"
if [[ ! -s "$TOKENS" ]]; then
  echo "ERROR: missing $TOKENS (run run_projlm_export.sbatch first)"
  exit 1
fi

# Fit banks on CPU if missing (mesh bank carries the random controls too).
mkdir -p "$BANKS"
MESH_BANK="$BANKS/${MODEL}_mesh_bank.npz"
TEX_BANK="$BANKS/${MODEL}_texture_bank.npz"
[[ -s "$MESH_BANK" ]] || python playgrounds/intervene.py --fit "$TOKENS" --target mesh --out "$MESH_BANK"
[[ -s "$TEX_BANK" ]] || python playgrounds/intervene.py --fit "$TOKENS" --target texture --out "$TEX_BANK"

case "$ARM" in
  shape)   BANK="$MESH_BANK"; ARM_FLAG=target ;;
  texture) BANK="$TEX_BANK";  ARM_FLAG=target ;;
  rand*)   BANK="$MESH_BANK"; ARM_FLAG="random${ARM#rand}" ;;
esac

SESSION="results/model.results/session_full_grid_v1a_erase_${ARM}"
mkdir -p "$SESSION"

ANY_FAIL=0
for COND in "${CONDITIONS[@]}"; do
  WORD_ARGS=()
  SUFFIX=""
  if [[ "$COND" == noun_label* ]]; then
    WORD_ARGS=(--word "$FIXED_WORD")
    SUFFIX="_${FIXED_WORD}"
  fi

  OUT="$SESSION/${MODEL}__${COND}${SUFFIX}.csv"
  OUT_ARGS=(-o "$OUT")
  if [[ -s "$OUT" ]]; then
    OUT_ARGS=(--resume "$OUT")
    echo "----- $COND: resuming $OUT ($(( $(wc -l < "$OUT") - 1 )) rows present) -----"
  else
    echo "----- $COND -> $OUT -----"
  fi

  START=$(date +%s)
  set +e
  python scripts/run_local.py \
    --models "$MODEL" \
    --ordering both \
    --prompt-condition "$COND" \
    --grid-pkg "$GRID_PKG" \
    --stim-set "$STIM_SET" \
    --intervene-bank "$BANK" \
    --intervene-arm "$ARM_FLAG" \
    --intervene-alpha 0.0 \
    "${WORD_ARGS[@]}" \
    "${OUT_ARGS[@]}"
  rc=$?
  set -e
  END=$(date +%s)

  if [[ $rc -ne 0 ]]; then
    echo "WARN: $MODEL $COND ($ARM) exited $rc after $((END-START))s (continuing)"
    ANY_FAIL=1
    continue
  fi
  ROWS=$(( $(wc -l < "$OUT") - 1 ))
  echo "OK $MODEL $COND $ARM rows=$ROWS elapsed_s=$((END-START))"
done

echo "DONE $MODEL $ARM $(date -Is) ANY_FAIL=$ANY_FAIL"
exit "$ANY_FAIL"
