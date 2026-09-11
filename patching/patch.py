"""Activation patching with KL divergence, replicating RomanLens Figure 5.

THEIR PROTOCOL
--------------
Build two prompts that differ ONLY in the script of the source word: one giving
the concept in native script (दरवाज़ा), one romanised (darwaza). Both translate
into a fixed third language -- they used Italian, a Latin-script language that
is neither input -- so the output side never varies. Patch the residual stream
from the source run into a receiving run at each layer, and read the
probability of the concept.

The summary statistic is the KL divergence between the native-source and
romanised-source next-token distributions. They report KL < 0.01 over 200
samples on gemma-2-9b-it, concluding the concept is encoded near-identically
regardless of the script it arrived in.

WHY
---------------------------------
Their result was on a tokenizer that fragmented Indic script heavily. Under
Gemma 3 native script costs FEWER tokens than romanisation. If script
invariance were partly an artifact of both forms being fragmented, a tokenizer
that handles native script cleanly could break it. If KL stays below 0.01
anyway, the invariance is a property of the representation rather than of the
tokenization -- a stronger claim than they could make.

PAIR BUILDERS
-------------
`script_pairs` reproduces their design. `swap_pairs` is a different question:
patch between a consistent prompt (demos and tag agree) and a conflict prompt
(demos say one language, tag says another) to find the layer at which the
explicit tag overrides the in-context demonstrations. That is task-
representation localisation rather than script invariance, but the code is the same.

"""

import json
import os

import numpy as np
import torch

from utils.models import find_backbone, find_softcap, unembed_fp32
from utils.vocab import build_token_sets


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def script_pairs(words, langs=("hi", "ta", "ml"), bridge="it",
                 bridge_tag="Italiano", n_demos=4, max_items=50):
    """RomanLens design: native-source vs romanised-source, fixed third target.

    The bridge language only ever appears as a TAG -- we never need its
    vocabulary, because what is scored is the concept in the source language
    and in English, not a correct Italian answer.
    """
    from promptutils.classes import TAG, surface

    out = []
    for lang in langs:
        pool = [w for w in words if w.get(lang)]
        if len(pool) <= n_demos:
            continue
        for i, t in enumerate(pool[:max_items]):
            demos = [w for j, w in enumerate(pool) if j != i][:n_demos]
            built = {}
            for cond, src in (("native", lang), ("roman", f"{lang}_rom")):
                lines = [f'{TAG[src]}: "{surface(d, src)}" - {bridge_tag}: ""'
                         for d in demos]
                lines.append(f'{TAG[src]}: "{surface(t, src)}" - '
                             f'{bridge_tag}: "')
                built[cond] = "\n".join(lines)
            out.append({
                "id": f"patch_{lang}_{t['en']}",
                "language": lang, "pair": "script",
                "prompt_a": built["native"], "prompt_b": built["roman"],
                "answers": {"en": t["en"],
                            f"{lang}_native": t[lang][0],
                            f"{lang}_roman": t[lang][1]}})
    return out


def swap_pairs(words, langs=("hi", "ta", "ml"), src_lang="fr", n_demos=3,
               max_items=50):
    """Consistent vs conflicting prompt: where does the tag beat the demos?"""
    from promptutils.classes import SWAP, TAG, surface

    out = []
    for lang in langs:
        final = SWAP[lang]
        pool = [w for w in words if (w.get(src_lang) or "").strip()]
        if len(pool) <= n_demos:
            continue
        for i, t in enumerate(pool[:max_items]):
            demos = [w for j, w in enumerate(pool) if j != i][:n_demos]
            head = [f'{TAG[src_lang]}: "{surface(d, src_lang)}" - '
                    f'{TAG[lang]}: "{surface(d, lang)}"' for d in demos]
            tail = f'{TAG[src_lang]}: "{surface(t, src_lang)}" - '
            out.append({
                "id": f"swap_{lang}to{final}_{t['en']}",
                "language": final, "demo_language": lang, "pair": "swap",
                "prompt_a": "\n".join(head + [tail + f'{TAG[lang]}: "']),
                "prompt_b": "\n".join(head + [tail + f'{TAG[final]}: "']),
                "answers": {"en": t["en"],
                            f"{lang}_native": t[lang][0],
                            f"{final}_native": t[final][0]}})
    return out


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------

@torch.no_grad()
def _final_probs(model, tokenizer, text, W, cap, device):
    enc = tokenizer(text, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1].float()
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return torch.softmax(logits, dim=-1).cpu()


@torch.no_grad()
def _cache_resid(model, backbone, tokenizer, text, device):
    """resid_post at the final token for every layer."""
    store = {}
    handles = []

    def make(i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            store[i] = h[:, -1, :].detach().clone()
        return hook

    for i, L in enumerate(backbone.layers):
        handles.append(L.register_forward_hook(make(i)))
    try:
        model(**tokenizer(text, return_tensors="pt").to(device))
    finally:
        for h in handles:
            h.remove()
    return store


@torch.no_grad()
def _patched_probs(model, backbone, tokenizer, text, donor, layer, W, cap,
                   device):
    """Run `text`, overwriting the final-token residual at `layer` with donor."""
    handle = None

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[:, -1, :] = donor.to(h.dtype)
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = backbone.layers[layer].register_forward_hook(hook)
    try:
        enc = tokenizer(text, return_tensors="pt").to(device)
        logits = model(**enc).logits[0, -1].float()
    finally:
        handle.remove()
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return torch.softmax(logits, dim=-1).cpu()


def kl(p, q, eps=1e-12):
    """KL(p || q) in nats, over the full vocabulary."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return float((p * (p.log() - q.log())).sum())


def run_patching(session, pairs, out_dir, max_items=None, direction="a_to_b"):
    """Patch A's residual into B at every layer; record KL and label curves.

    Two numbers per layer:
      kl_patched   KL(clean_B || patched_B). How much does importing A's state
                   at this layer change B's prediction? Near zero means the two
                   runs already agree there.
      kl_clean     KL(clean_A || clean_B), constant across layers, the baseline
                   difference between the two prompts with no intervention.

    RomanLens report the second. The first localises WHERE any difference lives.
    """
    os.makedirs(out_dir, exist_ok=True)
    model, tokenizer = session.model, session.tokenizer
    backbone, W, cap = session.backbone, session.W, session.cap
    n_layers, device = session.n_layers, session.device
    index = session.index

    pairs = pairs[:max_items] if max_items else pairs
    results = []

    for k, item in enumerate(pairs):
        a, b = item["prompt_a"], item["prompt_b"]
        if direction == "b_to_a":
            a, b = b, a
        ts = build_token_sets(index, item["answers"])

        clean_a = _final_probs(model, tokenizer, a, W, cap, device)
        clean_b = _final_probs(model, tokenizer, b, W, cap, device)
        donors = _cache_resid(model, backbone, tokenizer, a, device)

        rec = {"id": item["id"], "language": item["language"],
               "pair": item.get("pair", "script"),
               "kl_clean": kl(clean_a, clean_b),
               "kl_patched": [], "labels": {lab: [] for lab in ts}}
        for lab, ids in ts.items():
            rec["labels"][lab] = []
        for l in range(n_layers):
            p = _patched_probs(model, backbone, tokenizer, b,
                               donors[l][:, :], l, W, cap, device)
            rec["kl_patched"].append(kl(clean_b, p))
            for lab, ids in ts.items():
                rec["labels"][lab].append(float(p[torch.tensor(ids)].sum()))
        results.append(rec)
        if k % 10 == 0:
            print(f"[patch] {k}/{len(pairs)}  {item['id']}  "
                  f"kl_clean={rec['kl_clean']:.4f}")

    json.dump(results, open(f"{out_dir}/patching.json", "w"),
              ensure_ascii=False, indent=2)

    klc = np.array([r["kl_clean"] for r in results])
    klp = np.stack([r["kl_patched"] for r in results])
    print(f"\n[patch] {len(results)} pairs, {n_layers} layers")
    print(f"[patch] KL(clean_A || clean_B): mean {klc.mean():.5f}  "
          f"median {np.median(klc):.5f}  max {klc.max():.5f}")
    print(f"[patch] RomanLens report < 0.01 for the script pair on "
          f"gemma-2-9b-it")
    worst = int(klp.mean(0).argmax())
    print(f"[patch] patching changes B most at layer {worst} "
          f"(mean KL {klp.mean(0)[worst]:.4f})")
    return results