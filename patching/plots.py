"""Figures for the patching results."""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def report(out_dir, romanlens_threshold=0.01):
    path = os.path.join(out_dir, "patching.json")
    if not os.path.exists(path):
        raise SystemExit(f"no patching.json in {out_dir}")
    res = json.load(open(path))
    figs = os.path.join(out_dir, "figures")
    os.makedirs(figs, exist_ok=True)

    by_lang = defaultdict(list)
    for r in res:
        by_lang[r["language"]].append(r)

    # --- KL(clean_A || clean_B) per language ------------------------------
    langs = sorted(by_lang)
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(langs)), 3.6))
    data = [[r["kl_clean"] for r in by_lang[lg]] for lg in langs]
    ax.boxplot(data, labels=langs, showfliers=True)
    ax.axhline(romanlens_threshold, color="#c0392b", ls="--", lw=1.3,
               label=f"RomanLens < {romanlens_threshold}")
    ax.set_yscale("log")
    ax.set_ylabel("KL(clean A || clean B), nats")
    ax.set_title("distribution difference between the two prompts")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{figs}/kl_clean.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {figs}/kl_clean.png")

    # --- KL after patching, per layer -------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for lg in langs:
        a = np.stack([r["kl_patched"] for r in by_lang[lg]])
        m = a.mean(0)
        ci = 1.96 * a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0
        x = np.arange(len(m))
        ax.plot(x, m, lw=1.8, label=f"{lg} (n={len(a)})")
        ax.fill_between(x, m - ci, m + ci, alpha=0.15)
    ax.set_xlabel("patched layer")
    ax.set_ylabel("KL(clean B || patched B), nats")
    ax.set_title("where importing A's state changes B's prediction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(f"{figs}/kl_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {figs}/kl_by_layer.png")

    # --- label probabilities under patching -------------------------------
    fig, axes = plt.subplots(1, len(langs), squeeze=False,
                             figsize=(4.2 * len(langs), 3.6), sharey=True)
    for j, lg in enumerate(langs):
        a = axes[0][j]
        labels = defaultdict(list)
        for r in by_lang[lg]:
            for lab, arr in r["labels"].items():
                labels[lab].append(arr)
        for lab, arrs in sorted(labels.items()):
            m = np.stack(arrs).mean(0)
            a.plot(np.arange(len(m)), m, lw=1.6, label=lab)
        a.set_title(lg, fontsize=10)
        a.set_xlabel("patched layer")
        a.grid(alpha=0.25, lw=0.5)
        if j == 0:
            a.set_ylabel("P(label)")
            a.legend(fontsize=7)
    fig.suptitle("label probability under patching", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{figs}/labels_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {figs}/labels_by_layer.png")

    # --- table -------------------------------------------------------------
    lines = ["| lang | n | KL_clean mean | median | max | below 0.01 |",
             "|---|---|---|---|---|---|"]
    for lg in langs:
        k = np.array([r["kl_clean"] for r in by_lang[lg]])
        lines.append(f"| {lg} | {len(k)} | {k.mean():.5f} | "
                     f"{np.median(k):.5f} | {k.max():.5f} | "
                     f"{100 * (k < romanlens_threshold).mean():.0f}% |")
    table = "\n".join(lines)
    open(f"{out_dir}/summary.md", "w").write(table + "\n")
    print("\n" + table)
    return res