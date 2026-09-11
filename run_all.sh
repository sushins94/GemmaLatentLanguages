#!/usr/bin/env bash
# Full sweep. All python calls use python3.
#
#     bash run_all.sh                  everything
#     LIMIT=40 bash run_all.sh         smoke pass, ~40 prompts per set
#     SKIP_JLENS=1 bash run_all.sh     logit lens only
#     SKIP_PATCH=1 bash run_all.sh
#
# Results:  outputs/<wordlist>/<model>/<class>_<n>shot/
set -euo pipefail

MODEL_DIR=../data/models
ROMANLENS=../Romanlens/llm_logit_lens/data/langs

# model:dtype. 12B-class models do not fit in float32 on a 48 GB card.
# Naming differs across generations: Gemma 3 base is suffixed -pt, Gemma 4 base
# has no suffix. The PT/IT pairing strips "-it" rather than assuming "-pt".
MODELS=(gemma-3-4b-pt:float32
        gemma-3-4b-it:float32
        gemma-3-12b-pt:bfloat16
        gemma-3-12b-it:bfloat16)
#       gemma-4-12B:bfloat16
#       gemma-4-12B-it:bfloat16

SHOTS="1 2 3 4"
# Corpora to fit a Jacobian lens on. Fitting on more than one is the point:
# no published work does, so an English-fit transport cannot be distinguished
# from an English-biased model.
LENS_CORPORA=(en indic mixed)

LIMIT="${LIMIT:-}"
SKIP_PLACEHOLDER="${SKIP_PLACEHOLDER:-}"
SKIP_JLENS="${SKIP_JLENS:-}"
SKIP_PATCH="${SKIP_PATCH:-}"
GATE_MODEL="$MODEL_DIR/${MODELS[0]%%:*}"

limit_arg=()
[ -n "$LIMIT" ] && limit_arg=(--limit "$LIMIT")

# ---------------------------------------------------------------------------
# 0. word list, controls, prompt sets
# ---------------------------------------------------------------------------
# --model on build_words is optional: it only enables the validation report
# (fertility, single-token counts, collision survival per language). Worth
# having before spending GPU hours.
echo "=== word list ==="
python3 build_words.py --romanlens "$ROMANLENS" --model "$GATE_MODEL"
python3 build_controls.py --model "$GATE_MODEL" || \
    echo "[warn] build_controls failed (pip install wordfreq); falling back"

echo
echo "=== prompt sets ==="
python3 run.py sets --out sets_romanlens/ --shots $SHOTS
if [ -z "$SKIP_PLACEHOLDER" ]; then
  LLAT_WORDS=placeholder python3 run.py sets --out sets_placeholder/ --shots $SHOTS
fi

# ---------------------------------------------------------------------------
# 1. corpora + Jacobian lenses, one set per model
# ---------------------------------------------------------------------------
if [ -z "$SKIP_JLENS" ]; then
  echo
  echo "=== corpora ==="
  [ -f corpora/en.txt ] || python3 run.py corpora --out corpora/
  mkdir -p lenses
  for spec in "${MODELS[@]}"; do
    m="${spec%%:*}"
    [ -d "$MODEL_DIR/$m" ] || continue
    for c in "${LENS_CORPORA[@]}"; do
      [ -f "corpora/$c.txt" ] || continue
      out="lenses/${m}_${c}.pt"
      [ -f "$out" ] && { echo "have $out"; continue; }
      echo
      echo "=== fitting jlens: $m on $c ==="
      python3 run.py fit-lens --model "$MODEL_DIR/$m" \
          --corpus "corpora/$c.txt" --out "$out" --name "jlens_$c"
    done
  done
fi

# ---------------------------------------------------------------------------
# 2. the sweep -- one model load per (wordlist, model), all lenses scored from
#    the same capture
# ---------------------------------------------------------------------------
WORDLISTS=(romanlens)
[ -z "$SKIP_PLACEHOLDER" ] && WORDLISTS+=(placeholder)

for wl in "${WORDLISTS[@]}"; do
  sets_dir="sets_${wl}"
  [ -d "$sets_dir" ] || continue
  for spec in "${MODELS[@]}"; do
    m="${spec%%:*}"; dt="${spec##*:}"
    [ -d "$MODEL_DIR/$m" ] || { echo "skip $m (not found)"; continue; }

    lens_args=(logit)
    if [ -z "$SKIP_JLENS" ]; then
      for c in "${LENS_CORPORA[@]}"; do
        [ -f "lenses/${m}_${c}.pt" ] && lens_args+=("jlens_${c}=lenses/${m}_${c}.pt")
      done
    fi

    echo
    echo "=== $wl / $m ($dt)  lenses: ${lens_args[*]} ==="
    python3 run.py analyse --model "$MODEL_DIR/$m" \
        --prompts "$sets_dir"/*.json --wordlist "$wl" --dtype "$dt" \
        --lens "${lens_args[@]}" "${limit_arg[@]}"
  done
done

# ---------------------------------------------------------------------------
# 3. PT vs IT overlays (CPU only)
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
# 4. activation patching (RomanLens replication)
# ---------------------------------------------------------------------------
if [ -z "$SKIP_PATCH" ]; then
  echo
  echo "=== activation patching ==="
  for spec in "${MODELS[@]}"; do
    m="${spec%%:*}"; dt="${spec##*:}"
    [ -d "$MODEL_DIR/$m" ] || continue
    python3 run.py patch --model "$MODEL_DIR/$m" --out "patchruns/$m/script" \
        --pairs script --dtype "$dt"
  done
fi

echo
echo "figures:  outputs/*/*/*/figures/   patchruns/*/*/figures/"
echo "tables:   outputs/*/*/*/summary.md"