#!/usr/bin/env bash
# Resume after lens fitting step went OOM. Skips the word list, controls, prompt sets,
# corpora and Jacobian fitting, and uses WHATEVER lenses already exist on disk.
#
#     bash run_all_resume.sh
#     LIMIT=40 bash run_all_resume.sh      smoke pass
#     DRY=1   bash run_all_resume.sh       show the plan, run nothing
#
# Written because a 12B fit ran out of memory partway through. Lens coverage is
# therefore uneven across models: see the plan it prints before it starts.
# The logit lens is available for every model regardless, so every
# logit-lens result is complete and comparable; only the J-lens is partial.
#

set -euo pipefail

MODEL_DIR=../../data/models
MODELS=(gemma-3-4b-pt:float32
        gemma-3-4b-it:float32
        gemma-3-12b-pt:bfloat16
        gemma-3-12b-it:bfloat16)

LENS_CORPORA=(en indic mixed)

LIMIT="${LIMIT:-}"
DRY="${DRY:-}"
SKIP_PLACEHOLDER="${SKIP_PLACEHOLDER:-}"
SKIP_SLIDES="${SKIP_SLIDES:-}"

limit_arg=()
[ -n "$LIMIT" ] && limit_arg=(--limit "$LIMIT")

WORDLISTS=(romanlens)
[ -z "$SKIP_PLACEHOLDER" ] && WORDLISTS+=(placeholder)

# ---------------------------------------------------------------------------
# 0. preconditions and plan
# ---------------------------------------------------------------------------
for wl in "${WORDLISTS[@]}"; do
  [ -d "sets_${wl}" ] || { echo "missing sets_${wl}/ -- run: python3 run.py sets --out sets_${wl}/ --shots 1 2 3 4"; exit 1; }
done

echo "=== lens coverage ==="
printf "%-18s %-10s %s\n" MODEL DTYPE LENSES
for spec in "${MODELS[@]}"; do
  m="${spec%%:*}"; dt="${spec##*:}"
  if [ ! -d "$MODEL_DIR/$m" ]; then
    printf "%-18s %-10s %s\n" "$m" "$dt" "(model not found, will skip)"
    continue
  fi
  found="logit"
  for c in "${LENS_CORPORA[@]}"; do
    [ -f "lenses/${m}_${c}.pt" ] && found="$found jlens_${c}"
  done
  printf "%-18s %-10s %s\n" "$m" "$dt" "$found"
done
echo
echo "word lists: ${WORDLISTS[*]}"
echo "prompt sets per list: $(ls sets_${WORDLISTS[0]}/*.json 2>/dev/null | wc -l)"
[ -n "$LIMIT" ] && echo "LIMIT=$LIMIT prompts per set"
echo
[ -n "$DRY" ] && { echo "DRY=1, stopping here."; exit 0; }

# ---------------------------------------------------------------------------
# 1. the sweep -- one model load per (wordlist, model); every lens that exists
#    is scored from the SAME capture, so a missing lens costs nothing extra and
#    an extra lens costs no forward passes
# ---------------------------------------------------------------------------
for wl in "${WORDLISTS[@]}"; do
  for spec in "${MODELS[@]}"; do
    m="${spec%%:*}"; dt="${spec##*:}"
    [ -d "$MODEL_DIR/$m" ] || { echo "skip $m (not found)"; continue; }

    lens_args=(logit)
    for c in "${LENS_CORPORA[@]}"; do
      [ -f "lenses/${m}_${c}.pt" ] && lens_args+=("jlens_${c}=lenses/${m}_${c}.pt")
    done

    echo
    echo "=== $wl / $m ($dt)  lenses: ${lens_args[*]} ==="
    python3 run.py analyse --model "$MODEL_DIR/$m" \
        --prompts "sets_${wl}"/*.json --wordlist "$wl" --dtype "$dt" \
        --lens "${lens_args[@]}" "${limit_arg[@]}"
  done
done

# ---------------------------------------------------------------------------
# 2. PT vs IT overlays (CPU only)
#    report() uses the FIRST lens for the overlay, which is always logit, so
#    these comparisons are logit-vs-logit even where lens coverage differs.
# ---------------------------------------------------------------------------
echo
echo "=== PT vs IT overlays ==="
for wl in "${WORDLISTS[@]}"; do
  for itdir in outputs/"$wl"/*-it; do
    [ -d "$itdir" ] || continue
    base="${itdir%-it}"
    [ -d "$base" ] || base="${itdir%-it}-pt"
    [ -d "$base" ] || continue
    for p in "$base"/*/; do
      s=$(basename "$p")
      [ -d "$itdir/$s" ] && python3 run.py report --out "$p" --compare "$itdir/$s"
    done
  done
done

# ---------------------------------------------------------------------------
# 3. slides
# ---------------------------------------------------------------------------
if [ -z "$SKIP_SLIDES" ]; then
  echo
  echo "=== slides ==="
  python3 make_slides.py --root outputs --out slides/
fi

echo
echo "figures:  outputs/*/*/*/figures/"
echo "tables:   outputs/*/*/*/summary.md"

# ---------------------------------------------------------------------------
# If you later retry the 12B fits, the OOM was in the batched backward:
# is_grads_batched vmaps `chunk` cotangents of shape [chunk, 1, T, d_model]
# through one call, and at d_model=3840 that allocation plus the retained graph
# does not fit. Reduce the chunk, and optionally the prompt count:
#
#   python3 run.py fit-lens --model $MODEL_DIR/gemma-3-12b-pt \
#       --corpus corpora/indic.txt --out lenses/gemma-3-12b-pt_indic.pt \
#       --name jlens_indic --chunk 8 --n-prompts 10
#
# Anthropic's ablations show n=10 nearly matching n=1000, so lowering the
# prompt count is the cheaper lever and costs little fidelity. Re-running this
# script afterwards picks the new lens up automatically.
# ---------------------------------------------------------------------------
