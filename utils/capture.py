"""Capture residuals once, score them under any number of lenses.

RUN DIRECTORY
    outputs/<wordlist>/<model>/<class>_<n>shot/
        config.json           model, prompt set, layer count, lenses used
        records.json          the prompt set, annotated with kept / correct
        tokensets.json         vocab ids backing each label
        resid.npy             float32 [n_prompts, n_layers, d_model]
        entropy.npy           float32 [n_prompts, n_layers]
        completions.json      greedy continuation, for the accuracy gate
        curves__<lens>.npz    "<id>::<label>" -> P(label) per layer
        ranks__<lens>.npz     "<id>::<label>" -> rank of best token per layer
        figures/  summary.md

resid.npy is float32, NOT float16: Gemma's residual stream exceeds fp16's
65504 ceiling in later layers, so float16 storage silently produces inf.
"""

import datetime
import json
import os

import numpy as np
import torch

from .lenses import IdentityLens, lens_probs
from .models import (describe, find_backbone, find_softcap, load_model,
                     make_lm, unembed_fp32)
from .vocab import (VocabIndex, build_token_sets, collision_detail,
                    has_collision)

SCRIPT_LABELS = ("LATIN", "DEVANAGARI", "TAMIL", "MALAYALAM")


def default_out_dir(model_path, prompts_path, root="outputs", wordlist=None):
    m = os.path.basename(os.path.normpath(model_path)) or "model"
    s = os.path.splitext(os.path.basename(prompts_path))[0]
    parts = [root] + ([wordlist] if wordlist else []) + [m, s]
    return os.path.join(*parts)


@torch.no_grad()
def capture_residuals(lm, envoy, text, n_layers):
    """resid_post at the final token, every layer, unnormed.

    Do NOT rewrite as a comprehension: a comprehension has its own scope and
    the .save() proxies do not survive it.
    """
    saved = []
    with lm.trace(text):
        for l in range(n_layers):
            saved.append(envoy.layers[l].output[0][0, -1, :].save())
    return torch.stack([s.detach().float() for s in saved])


def first_answer(s):
    """Cut a generation back to the answer span.

    Few-shot prompts end on an open quote, so a correct completion looks like
    `பூ"\\nFrançais: ...` -- the model closes the quote and continues the
    pattern, which is expected. Without this cut the first whitespace token is
    `பூ"` and exact match fails on a correct answer.
    """
    for stop in ('"', "\u201d", "\n"):
        i = s.find(stop)
        if i > 0:
            s = s[:i]
    return s


def normalise(s):
    """Light normalisation before matching. Does not fix the deeper problem:
    exact match undercounts morphologically rich languages more than Hindi, so
    a resource gradient in acc_exact may be a scoring artifact. acc_tok1 is the
    cross-check."""
    return s.strip().strip('"\u201c\u201d\u2018\u2019\'.,;:!?()[]').casefold()


def first_token_correct(top1_id, want, tokenizer):
    """Does the top token equal the first token of the answer?

    This is the metric that matches the curves: P(lang) sums over tokens that
    could BEGIN the answer. A gap between this and acc_exact means the model
    reached the right concept but could not spell the rest.

    BOTH variants are accepted. Encoding only " "+want gives the SentencePiece
    space-prefixed token, but few-shot prompts end on an open quote so the real
    next token usually has no leading space -- comparing against the
    space-prefixed form alone made this metric read zero almost everywhere.
    Start(w) has always handled both variants, which is why P(lang) was
    unaffected.
    """
    if not want:
        return None
    first = set()
    for variant in (" " + want, want):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if ids:
            first.add(ids[0])
    return top1_id in first if first else None


def best_rank(probs, ids):
    """Best (lowest) rank among `ids`, per layer. 0 = top of the distribution.

    A rank is just how many tokens beat this one, so it needs a comparison and
    a sum -- not a sort. Full-sorting the 262k-wide distribution per layer per
    lens (the earlier implementation) was the single largest cost in capture.

    The minimum rank over a set is the rank of the set's most probable member,
    so one comparison against that maximum suffices.
    """
    pmax = probs[:, ids].max(-1, keepdim=True).values
    return (probs > pmax).sum(-1)


def score(probs, token_sets, rid, curves, ranks, script_ids=None):
    """P(label) and rank per layer, plus whole-script mass.

    `probs` stays on the GPU; only the small per-layer results come back.
    Script mass needs no answer key, so it is defined for gibberish too, and it
    is the only transition measure comparable across every prompt class.
    """
    dev = probs.device
    for label, ids in token_sets.items():
        t = torch.as_tensor(ids, device=dev)
        curves[f"{rid}::{label}"] = probs[:, t].sum(-1).cpu().numpy()
        ranks[f"{rid}::{label}"] = best_rank(probs, t).cpu().numpy()
    for scr, ids in (script_ids or {}).items():
        t = ids.to(dev)
        curves[f"{rid}::script_{scr}"] = probs[:, t].sum(-1).cpu().numpy()


class Session:
    """A loaded model plus everything derived from it, reused across sets.

    Loading costs ~20 s and the 262k-token VocabIndex another ~30 s. Both are
    per-MODEL, so a session pays that once and every later set is forward
    passes only.
    """

    def __init__(self, model_path, device="cuda", dtype=torch.float32):
        from transformers import AutoTokenizer

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = load_model(model_path, device=device, dtype=dtype)
        self.model.eval()
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = \
                self.tokenizer.eos_token_id

        self.info = describe(self.model, self.tokenizer)
        self.path, self.backbone, self.n_layers = find_backbone(self.model)
        self.W = unembed_fp32(self.model)
        self.cap = find_softcap(self.model)
        self.lm, self.envoy = make_lm(self.model, self.tokenizer, self.path)
        self.dtype = dtype
        self.device = device

        print("[vocab] indexing (once per model) ...")
        self.index = VocabIndex(self.tokenizer)
        self.script_ids = {s: torch.tensor(self.index.by_script[s])
                           for s in SCRIPT_LABELS if self.index.by_script.get(s)}

    def free(self):
        del self.model, self.lm, self.envoy
        torch.cuda.empty_cache()


def capture_set(session, prompts_path, out_dir=None, lenses=None, chat=False,
                limit=None, keep_collisions=False, max_new_tokens=6,
                root="outputs", wordlist=None):
    """Capture one prompt set and score it under every lens supplied."""
    lenses = lenses or [IdentityLens()]
    tokenizer, model = session.tokenizer, session.model
    backbone, W, cap = session.backbone, session.W, session.cap
    n_layers, index = session.n_layers, session.index

    out_dir = out_dir or default_out_dir(session.model_path, prompts_path,
                                         root, wordlist)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[run] {out_dir}")

    records = json.load(open(prompts_path))
    if limit:
        records = records[:limit]

    token_sets, dropped = {}, 0
    for r in records:
        ts = build_token_sets(index, r["answers"])
        r["collision"] = has_collision(ts, r.get("collision_labels"))
        r["kept"] = keep_collisions or not r["collision"]
        dropped += int(r["collision"])
        token_sets[r["id"]] = ts
    print(f"[vocab] {dropped}/{len(records)} items collide "
          f"({'kept' if keep_collisions else 'excluded'})")
    if records and dropped / len(records) > 0.5:
        print("[vocab] WARNING: most items dropped. Shared tokens:")
        shown = 0
        for r in records:
            if not r["collision"] or shown >= 3:
                continue
            for a, b, toks in collision_detail(token_sets[r["id"]], index,
                                               r.get("collision_labels")):
                print(f"   {r['id']}: {a} vs {b} share {toks}")
            shown += 1

    resid = np.zeros((len(records), n_layers, session.info["d_model"]),
                     dtype=np.float32)
    entropy = np.zeros((len(records), n_layers), dtype=np.float32)
    per_lens = {ln.name: ({}, {}) for ln in lenses}
    completions = []

    for i, r in enumerate(records):
        text = r["prompt"]
        if chat:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True)

        h = capture_residuals(session.lm, session.envoy, text, n_layers)
        resid[i] = h.cpu().numpy()

        for ln in lenses:
            probs = lens_probs(h, ln, backbone, W, cap, device=session.device)
            if isinstance(ln, IdentityLens):
                entropy[i] = (-(probs * torch.log(probs + 1e-30)).sum(-1)
                              .cpu().numpy())
                top1 = int(probs[-1].argmax())
            c, k = per_lens[ln.name]
            score(probs, token_sets[r["id"]], r["id"], c, k,
                  session.script_ids)

        enc = tokenizer(text, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False)
        out = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=True)
        want = r["answers"].get(r.get("target_label", ""), "")
        nw, no = normalise(want), normalise(first_answer(out))
        r["correct"] = bool(want) and (nw in no)
        r["correct_exact"] = bool(want) and no.split()[:1] == [nw]
        r["correct_first_token"] = first_token_correct(top1, want, tokenizer)
        completions.append({"id": r["id"], "completion": out,
                            "answer_span": first_answer(out),
                            "expected": want, "correct": r["correct"],
                            "correct_exact": r["correct_exact"],
                            "correct_first_token": r["correct_first_token"]})

        if i % 50 == 0:
            print(f"[run] {i}/{len(records)}  {r['id']}")

    np.save(f"{out_dir}/resid.npy", resid)
    np.save(f"{out_dir}/entropy.npy", entropy)
    for name, (c, k) in per_lens.items():
        np.savez(f"{out_dir}/curves__{name}.npz", **c)
        np.savez(f"{out_dir}/ranks__{name}.npz", **k)
    for name, obj in (("records", records), ("completions", completions),
                      ("tokensets", token_sets)):
        json.dump(obj, open(f"{out_dir}/{name}.json", "w"),
                  ensure_ascii=False, indent=2)
    json.dump({**session.info, "model": os.path.abspath(session.model_path),
               "prompt_set": os.path.abspath(prompts_path),
               "weight_dtype": str(session.dtype).replace("torch.", ""),
               "lens_dtype": "float32",
               "lenses": [ln.meta for ln in lenses],
               "chat_template": chat, "n_prompts": len(records),
               "n_collisions": dropped,
               "timestamp": datetime.datetime.now().isoformat(timespec="seconds")},
              open(f"{out_dir}/config.json", "w"), indent=2, default=str)

    kept = [r for r in records if r["kept"]]
    print(f"\n[gate] kept {len(kept)}/{len(records)}")
    print(f"   {'lang':>4} {'script':>6} {'n':>4}  {'sub':>5} {'exact':>5} {'tok1':>5}")
    for lang in sorted({r["language"] for r in kept}):
        for script in sorted({r["script"] for r in kept}):
            sub = [r for r in kept
                   if r["language"] == lang and r["script"] == script]
            if not sub:
                continue
            ft = [r["correct_first_token"] for r in sub
                  if r["correct_first_token"] is not None]
            print(f"   {lang:>4} {script:>6} {len(sub):>4}  "
                  f"{np.mean([r['correct'] for r in sub]):>5.3f} "
                  f"{np.mean([r['correct_exact'] for r in sub]):>5.3f} "
                  f"{(np.mean(ft) if ft else float('nan')):>5.3f}")
    return out_dir