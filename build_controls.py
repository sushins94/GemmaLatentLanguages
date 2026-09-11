#!/usr/bin/env python3
"""
Build frequency-matched English control words.

    pip install wordfreq
    python3 build_controls.py --model ../data/models/gemma-3-4b-pt/
    python3 run.py sets --out sets_romanlens/          # picks the file up

Writes data/controls.json, a mapping from each English word in the word list to
an unrelated English word matched on two properties.

WHY MATCH, AND ON WHAT
----------------------
The control asks whether mid-layer English mass is specific to the correct
English word or reflects a general pull toward English tokens. For that
comparison to mean anything, the control must be as *attractive* as the target
for reasons unrelated to meaning. Two such reasons:

  unigram frequency   frequent tokens receive more gradient during training and
                      end up with larger unembedding rows. Row norm is exactly
                      the quantity control D tests, so an unmatched control
                      leaves that confound inside the comparison meant to
                      isolate it.

  |Start(w)|          P(lang) sums over every token that could begin the word.
                      A word the tokenizer splits more finely has a larger set
                      and collects more mass for purely mechanical reasons.

Matching on both makes a low control curve informative: the control was equally
easy to produce and still did not appear.

NOTES:
--------------------------
Semantic relatedness. A control drawn at random may happen to be associated
with the target ("river" for "water"), which would inflate it. The exclusion
list below removes the word list and its translations, but not general
association. 

The matched control is a fixed choice per word, so it shares the
target's frequency but not its context. It answers "would an equally common
English word appear here", not "would any English word appear here".
"""

import argparse
import json
import os
from collections import Counter

DEFAULT_POOL_SIZE = 40000
MAX_REUSE = 3


def load_frequencies(pool_size):
    """Candidate English words with Zipf frequencies, most frequent first."""
    try:
        from wordfreq import top_n_list, zipf_frequency
    except ImportError:
        raise SystemExit("pip install wordfreq")
    words = [w for w in top_n_list("en", pool_size)
             if w.isalpha() and w.islower() and len(w) >= 3]
    return [(w, zipf_frequency(w, "en")) for w in words]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="tokenizer for Start(w)")
    ap.add_argument("--words", default="promptutils/words.json")
    ap.add_argument("--out", default="promptutils/controls.json")
    ap.add_argument("--pool", type=int, default=DEFAULT_POOL_SIZE)
    ap.add_argument("--max-reuse", type=int, default=MAX_REUSE)
    ap.add_argument("--tol-zipf", type=float, default=0.5,
                    help="acceptable Zipf gap before falling back to the "
                         "closest available candidate")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from utils.vocab import VocabIndex

    words = json.load(open(a.words, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(a.model)
    index = VocabIndex(tok)

    # everything appearing in the word list, in any language or script, is
    # excluded: a control must not be a translation of anything being measured
    banned = set()
    for w in words:
        banned.add(w["en"].lower())
        for k in ("fr", "de", "zh"):
            if w.get(k):
                banned.add(str(w[k]).lower())
        for lg in ("hi", "ta", "ml"):
            if lg in w:
                banned.update(str(x).lower() for x in w[lg])

    print(f"[pool] loading top {a.pool} English words ...")
    pool = [(w, f) for w, f in load_frequencies(a.pool) if w not in banned]

    from wordfreq import zipf_frequency
    start = {}
    nstart = {}
    for w, _ in pool:
        s = index.start_token_ids(w)
        start[w] = s
        nstart[w] = len(s)

    used = Counter()
    out, gaps = {}, []
    for item in words:
        en = item["en"]
        f_t = zipf_frequency(en, "en")
        n_t = len(index.start_token_ids(en))
        forbid = set(index.start_token_ids(en))
        for lg in ("hi", "ta", "ml"):
            if lg in item:
                for form in item[lg]:
                    forbid |= index.start_token_ids(form)

        best = None
        for c, f_c in pool:
            if used[c] >= a.max_reuse or nstart[c] != n_t:
                continue
            if start[c] & forbid:          # would trip the collision filter
                continue
            d = abs(f_c - f_t)
            if best is None or d < best[0]:
                best = (d, c, f_c)
                if d < 0.05:
                    break

        if best is None:                   # relax the exact-length constraint
            for c, f_c in pool:
                if used[c] >= a.max_reuse or start[c] & forbid:
                    continue
                d = abs(f_c - f_t) + 0.5 * abs(nstart[c] - n_t)
                if best is None or d < best[0]:
                    best = (d, c, f_c)

        if best is None:
            print(f"[warn] no control for {en!r}")
            continue
        d, c, f_c = best
        used[c] += 1
        gaps.append(abs(f_c - f_t))
        out[en] = {"control": c, "zipf_target": round(f_t, 2),
                   "zipf_control": round(f_c, 2),
                   "n_start_target": n_t, "n_start_control": nstart[c]}

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=2)

    gaps.sort()
    n = len(gaps)
    print(f"\n[wrote] {a.out}: {n} controls")
    if n:
        print(f"[match] Zipf gap  median {gaps[n // 2]:.2f}  "
              f"90th pct {gaps[int(0.9 * n)]:.2f}  max {gaps[-1]:.2f}")
        exact = sum(v["n_start_target"] == v["n_start_control"]
                    for v in out.values())
        print(f"[match] |Start(w)| matched exactly for {exact}/{n}")
        print(f"[match] distinct control words: {len(set(v['control'] for v in out.values()))}")
        print("\n[sample]")
        for en, v in list(out.items())[:8]:
            print(f"   {en:<12} -> {v['control']:<12} "
                  f"zipf {v['zipf_target']:.2f} vs {v['zipf_control']:.2f}  "
                  f"|Start| {v['n_start_target']} vs {v['n_start_control']}")
    print("\n[next] python3 run.py sets --out sets_romanlens/")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()