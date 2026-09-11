#!/usr/bin/env python3
"""
Driver.

    python3 run.py sets     --out sets/ --shots 1 2 3 4
    python3 run.py corpora  --out corpora/            # FLORES-200
    python3 run.py fit-lens --model M --corpus corpora/en.txt --out lenses/M_en.pt
    python3 run.py analyse  --model M --prompts sets/*.json --lens logit jlens=lenses/M_en.pt
    python3 run.py report   --out outputs/.../class1_translation_4shot
    python3 run.py patch    --model M --out patchruns/M

LENSES
    logit                       the ordinary logit lens (no transport)
    <tag>=<path.pt>             a fitted Jacobian lens
    <tag>=<path.pt>:<lam>       with shrinkage toward the identity

All lenses share one capture: residuals are read once and scored under each,
so adding a lens costs no extra forward passes.

LAYOUT
    outputs/<wordlist>/<model>/<class>_<n>shot/
"""

import argparse
import glob
import json
import os

# torch / transformers / nnsight are imported inside the commands that need
# them, so `sets`, `corpora` and `report` run on a login node with no GPU.


def cmd_sets(args):
    from promptutils import BUILDERS, SHOT_SWEPT
    from promptutils.classes import CONTROLS_MATCHED
    from promptutils.words import IS_PLACEHOLDER, WORDS

    if IS_PLACEHOLDER:
        print("=" * 70)
        print(f"PLACEHOLDER WORD LIST ({len(WORDS)} entries). Fine for "
              f"debugging; not for any reported number.")
        print("  python3 build_words.py --romanlens <path> --model <model>")
        print("=" * 70)
    print(f"[controls] en_control is "
          + ("frequency- and length-matched (promptutils/controls.json)"
             if CONTROLS_MATCHED else
             "an unmatched common noun; run build_controls.py to match"))

    os.makedirs(args.out, exist_ok=True)
    names = [args.only] if args.only else list(BUILDERS)
    for name in names:
        shots = args.shots if name in SHOT_SWEPT else [None]
        for n in shots:
            recs = BUILDERS[name]() if n is None else BUILDERS[name](n_demos=n)
            tag = f"{name}_{n}shot" if n is not None else name
            p = os.path.join(args.out, f"{tag}.json")
            json.dump(recs, open(p, "w"), ensure_ascii=False, indent=2)
            print(f"{p}: {len(recs)} prompts  "
                  f"langs={sorted({r['language'] for r in recs})}")


FLORES_CODE = {"en": "eng_Latn", "hi": "hin_Deva",
               "ta": "tam_Taml", "ml": "mal_Mlym"}


def cmd_corpora(args):
    """FLORES-200 corpora for Jacobian fitting.

    FLORES is N-WAY PARALLEL: en.txt and indic.txt hold the SAME sentences in
    different languages, so comparing lenses fitted on each is a controlled
    comparison rather than one confounded by content. No published J-lens work
    fits on more than one corpus, so this comparison is the open question.
    """
    import random
    from datasets import load_dataset

    os.makedirs(args.out, exist_ok=True)
    sents = {}
    for lg in args.langs:
        ds = None
        for repo in ("Muennighoff/flores200", "facebook/flores"):
            try:
                ds = load_dataset(repo, name=FLORES_CODE[lg], split=args.split)
                print(f"[flores] {lg} <- {repo} ({len(ds)} rows)")
                break
            except Exception as e:                       # noqa: BLE001
                print(f"[flores] {repo} failed for {lg}: {type(e).__name__}")
        if ds is None:
            raise SystemExit("could not load FLORES; pip install -U datasets")
        col = "sentence" if "sentence" in ds.column_names else ds.column_names[0]
        sents[lg] = [r[col] for r in ds]
    n = min(len(v) for v in sents.values())
    sents = {k: v[:n] for k, v in sents.items()}

    rng = random.Random(args.seed)

    def pack(pick):
        return [" ".join(pick(rng.randrange(n))
                         for _ in range(args.sents_per_doc)).replace("\n", " ")
                for _ in range(args.n_docs)]

    docs = {lg: pack(lambda i, r=rows: r[i]) for lg, rows in sents.items()}
    indic = [l for l in sents if l != "en"]
    if indic:
        docs["indic"] = pack(lambda i: sents[rng.choice(indic)][i])
    if len(sents) > 1:
        docs["mixed"] = pack(lambda i: sents[rng.choice(list(sents))][i])
    for name, d in docs.items():
        p = os.path.join(args.out, f"{name}.txt")
        open(p, "w", encoding="utf-8").write("\n".join(d) + "\n")
        print(f"[corpora] {p}: {len(d)} documents")


def cmd_fit_lens(args):
    import torch
    from transformers import AutoTokenizer
    from utils.lenses import fit_jacobian, finite_difference_check
    from utils.models import load_model

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, device="cuda", dtype=torch.bfloat16)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)          # gradients w.r.t. activations only

    texts = [l.strip() for l in open(args.corpus) if l.strip()]
    print(f"[fit] {len(texts)} texts available, using {args.n_prompts}")
    lens = fit_jacobian(model, tokenizer, texts, n_prompts=args.n_prompts,
                        seq_len=args.seq_len, skip_first=args.skip_first,
                        chunk=args.chunk, name=args.name or "jlens")
    lens._meta["corpus"] = os.path.abspath(args.corpus)
    lens._meta["model"] = os.path.abspath(args.model)
    lens.save(args.out)
    print(f"[fit] saved {args.out}")
    for l in (1, len(lens.J) // 2, len(lens.J) - 3):
        _, ratio = lens.spectrum(l)
        print(f"[spec] layer {l}: s_max/s_median = {ratio:.1f}")

    if not args.no_validate:
        del model
        torch.cuda.empty_cache()
        model = load_model(args.model, device="cuda", dtype=torch.float32)
        model.eval()
        print("\n[validate] finite-difference check "
              "(cosine below 0.3 means the Jacobian is wrong)")
        for l in (1, len(lens.J) // 3, 2 * len(lens.J) // 3, len(lens.J) - 3):
            r = finite_difference_check(model, lens, tokenizer, l)
            flag = "OK" if r["cosine"] > 0.3 else "TOO NOISY"
            print(f"  layer {l:>2}: cos {r['cosine']:+.3f} "
                  f"+/- {r['cosine_sd']:.3f}   "
                  f"norm ratio {r['norm_ratio']:.3f}   {flag}")


def cmd_analyse(args):
    """One model load covers every prompt set and every lens passed."""
    import torch
    from plotutils.plots import report
    from utils.capture import Session, capture_set
    from utils.lenses import make_lens

    lenses = [make_lens(s) for s in (args.lens or ["logit"])]
    print(f"[lens] {[l.name for l in lenses]}")

    session = Session(args.model, dtype=getattr(torch, args.dtype))
    try:
        for p in args.prompts:
            run_dir = capture_set(session, p, out_dir=args.out, lenses=lenses,
                                  chat=args.chat, limit=args.limit,
                                  keep_collisions=args.keep_collisions,
                                  root=args.root, wordlist=args.wordlist)
            report(run_dir, compare_run=args.compare)
    finally:
        session.free()


def cmd_report(args):
    from plotutils.plots import report
    report(args.out, lenses=args.lens or None, compare_run=args.compare)


def cmd_patch(args):
    import torch
    from patching.patch import run_patching, script_pairs, swap_pairs
    from patching.plots import report
    from promptutils.words import WORDS
    from utils.capture import Session

    builder = {"script": script_pairs, "swap": swap_pairs}[args.pairs]
    pairs = builder(WORDS, max_items=args.max_items)
    print(f"[patch] {len(pairs)} {args.pairs} pairs")

    session = Session(args.model, dtype=getattr(torch, args.dtype))
    try:
        run_patching(session, pairs, args.out, direction=args.direction)
    finally:
        session.free()
    report(args.out)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sets", help="write prompt-set JSON")
    s.add_argument("--out", default="sets")
    s.add_argument("--only", default=None)
    s.add_argument("--shots", type=int, nargs="*", default=[1, 2, 3, 4],
                   help="demonstration counts to sweep for the shot-swept "
                        "classes; others are written once")
    s.set_defaults(fn=cmd_sets)

    c = sub.add_parser("corpora", help="FLORES-200 corpora for lens fitting")
    c.add_argument("--out", default="corpora")
    c.add_argument("--langs", nargs="*", default=["en", "hi", "ta", "ml"])
    c.add_argument("--split", default="dev")
    c.add_argument("--n-docs", type=int, default=30)
    c.add_argument("--sents-per-doc", type=int, default=6)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(fn=cmd_corpora)

    f = sub.add_parser("fit-lens", help="fit a Jacobian lens")
    f.add_argument("--model", required=True)
    f.add_argument("--corpus", required=True, help="one document per line")
    f.add_argument("--out", required=True)
    f.add_argument("--name", default=None, help="lens tag, e.g. jlens_en")
    f.add_argument("--n-prompts", type=int, default=25)
    f.add_argument("--seq-len", type=int, default=128)
    f.add_argument("--skip-first", type=int, default=4)
    f.add_argument("--chunk", type=int, default=32)
    f.add_argument("--no-validate", action="store_true")
    f.set_defaults(fn=cmd_fit_lens)

    a = sub.add_parser("analyse", help="capture prompt sets and report")
    a.add_argument("--model", required=True)
    a.add_argument("--prompts", required=True, nargs="+")
    a.add_argument("--lens", nargs="*", default=None,
                   help="logit | tag=path.pt[:lam]; several are scored from "
                        "one capture")
    a.add_argument("--out", default=None)
    a.add_argument("--root", default="outputs")
    a.add_argument("--wordlist", default=None,
                   help="inserted as outputs/<wordlist>/<model>/<set>/")
    a.add_argument("--chat", action="store_true")
    a.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="weights only; the lens is always float32")
    a.add_argument("--limit", type=int, default=None)
    a.add_argument("--keep-collisions", action="store_true")
    a.add_argument("--compare", default=None)
    a.set_defaults(fn=cmd_analyse)

    r = sub.add_parser("report", help="regenerate figures and table")
    r.add_argument("--out", required=True)
    r.add_argument("--lens", nargs="*", default=None)
    r.add_argument("--compare", default=None)
    r.set_defaults(fn=cmd_report)

    p = sub.add_parser("patch", help="activation patching with KL divergence")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", default="script", choices=["script", "swap"],
                   help="'script' reproduces RomanLens (native vs romanised "
                        "source, fixed third target); 'swap' patches between "
                        "a consistent and a conflicting prompt")
    p.add_argument("--direction", default="a_to_b",
                   choices=["a_to_b", "b_to_a"])
    p.add_argument("--max-items", type=int, default=50)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"])
    p.set_defaults(fn=cmd_patch)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()