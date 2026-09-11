"""Figures. Grouping follows Wendler Figure 2: one figure per task, a grid with
scripts as rows and languages as columns, curves coloured by ROLE so a colour
means the same thing everywhere."""

import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .summary import (available_lenses, cell_stats, collect, collect_scripts,
                      load_run, mean_ci, per_prompt_script_switch,
                      per_prompt_switch, render_table, summary_rows)

COLOR = {"en": "#c0392b", "native": "#2471a3", "roman": "#f39c12",
         "en_ctrl": "#7f8c8d", "src": "#27ae60"}
STYLE = {"en": "-", "native": "-", "roman": "-", "en_ctrl": ":", "src": "--"}
ROLE_ORDER = ("en", "native", "roman", "en_ctrl", "src")
SCRIPT_COLOR = {"LATIN": "#c0392b", "DEVANAGARI": "#8e44ad",
                "TAMIL": "#2471a3", "MALAYALAM": "#16a085"}
METRICS = [("correct", "substring", "#34495e"),
           ("correct_exact", "exact", "#2471a3"),
           ("correct_first_token", "first token", "#f39c12")]


def save(fig, path, dpi=150):
    """tight_layout alone still clips a suptitle when the grid is dense or tick
    labels are rotated; bbox_inches measures the rendered figure instead."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  {path}")


def _grid(cells, draw, title, ylabel, path, logy=False):
    if not cells:
        return
    scripts = sorted({c[2] for c in cells})
    langs = sorted({c[1] for c in cells})
    fig, ax = plt.subplots(len(scripts), len(langs), squeeze=False,
                           figsize=(3.9 * len(langs), 3.3 * len(scripts)),
                           sharex=True, sharey=True)
    drew = False
    for i, sc in enumerate(scripts):
        for j, lg in enumerate(langs):
            a = ax[i][j]
            key = next((c for c in cells if c[1] == lg and c[2] == sc), None)
            n = draw(a, key) if key else 0
            if not n:
                a.set_visible(False)
                continue
            drew = True
            if logy:
                a.set_yscale("log")
            a.set_title(f"{lg} / {sc}  (n={n})", fontsize=10)
            a.grid(alpha=0.25, lw=0.5)
            if i == len(scripts) - 1:
                a.set_xlabel("layer")
            if j == 0:
                a.set_ylabel(ylabel)
    if not drew:
        plt.close(fig)
        return
    ax[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, path)


def curves_figure(by, task, title, path, logy=False, offset=0):
    cells = [c for c in by if c[0] == task]

    def draw(a, key):
        n = 0
        for ro in ROLE_ORDER:
            if ro not in by[key]:
                continue
            m, ci, n = mean_ci(by[key][ro])
            x = np.arange(len(m))
            a.plot(x, m + offset, color=COLOR[ro], ls=STYLE[ro], lw=1.8,
                   label=ro)
            if not logy:
                a.fill_between(x, m - ci, m + ci, color=COLOR[ro], alpha=0.15)
        return n

    _grid(cells, draw, title, "rank (1 = top)" if logy else "P(label)",
          path, logy=logy)


def scripts_figure(by, task, title, path):
    """Total probability mass on each Unicode script per layer.

    Works with no answer key, so it applies to gibberish too, and it is the
    control for script priming: a Latin language tag pushes the model toward
    Latin output regardless of meaning.
    """
    cells = [c for c in by if c[0] == task]

    def draw(a, key):
        n = 0
        for name, arrs in sorted(by[key].items()):
            m, ci, n = mean_ci(arrs)
            x = np.arange(len(m))
            a.plot(x, m, lw=1.7, label=name, color=SCRIPT_COLOR.get(name))
            a.fill_between(x, m - ci, m + ci, alpha=0.15,
                           color=SCRIPT_COLOR.get(name))
        return n

    _grid(cells, draw, title, "mass on script", path)


def switch_figure(by_rank, by_script, task, path, n_layers=None):
    """Distribution of the per-prompt switch layer.

    Read beside the averaged curves: a wide or split histogram means the mean
    curve is hiding structure, and a resource-level effect may live in the
    spread rather than the mean.
    """
    cells = sorted(k for k in by_rank if k[0] == task)
    if not cells:
        return
    scripts = sorted({c[2] for c in cells})
    langs = sorted({c[1] for c in cells})
    fig, ax = plt.subplots(len(scripts), len(langs), squeeze=False,
                           figsize=(3.9 * len(langs), 3.1 * len(scripts)),
                           sharex=True)
    drew, modes = False, set()
    for i, sc in enumerate(scripts):
        for j, lg in enumerate(langs):
            a = ax[i][j]
            cell = by_rank.get((task, lg, sc))
            sw = (per_prompt_switch(cell, "native" if sc == "native" else "roman")
                  if cell else np.array([]))
            mode = "target overtakes en"
            if not len(sw) and by_script:
                sw = per_prompt_script_switch(by_script.get((task, lg, sc), {}),
                                              lg)
                mode = "target script overtakes Latin"
            if not len(sw):
                a.set_visible(False)
                continue
            good, never = sw[sw >= 0], int((sw < 0).sum())
            if len(good):
                hi = n_layers or int(good.max()) + 2
                a.hist(good, bins=np.arange(-0.5, hi + 0.5, 1.0),
                       color="#2471a3", alpha=0.85)
                a.axvline(good.mean(), color="#c0392b", ls="--", lw=1.4,
                          label=f"mean {good.mean():.1f}")
                a.axvline(np.median(good), color="#f39c12", ls=":", lw=1.4,
                          label=f"median {np.median(good):.0f}")
                a.legend(fontsize=7)
            modes.add(mode)
            drew = True
            a.set_title(f"{lg} / {sc}  n={len(good)}"
                        + (f"  (+{never} never)" if never else ""), fontsize=9)
            a.grid(alpha=0.25, lw=0.5)
            if i == len(scripts) - 1:
                a.set_xlabel("switch layer")
            if j == 0:
                a.set_ylabel("prompts")
    if not drew:
        plt.close(fig)
        return
    fig.suptitle(f"per-prompt switch layer — {task}  ({'; '.join(sorted(modes))})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, path)


def accuracy_figure(records, path, title):
    """Three metrics side by side. acc_tok1 is the one that matches the curves,
    since P(lang) is a first-token quantity. A large gap between it and
    acc_exact means the model reached the right concept but could not spell the
    rest -- common in agglutinative languages, and not the same as failing."""
    kept = [r for r in records if r.get("kept", True)]
    cells = defaultdict(lambda: defaultdict(list))
    for r in kept:
        key = (r["task"], r["language"], r["script"])
        for m, _, _ in METRICS:
            if r.get(m) is not None:
                cells[key][m].append(bool(r[m]))
    keys = sorted(cells)
    if not keys:
        return
    width = 0.8 / len(METRICS)
    fig, ax = plt.subplots(figsize=(min(16.0, max(7.0, 0.85 * len(keys))),
                                    4.2 + 0.6 * (len(keys) > 12)))
    for i, (m, label, colour) in enumerate(METRICS):
        vals = [np.mean(cells[k][m]) if cells[k].get(m) else np.nan
                for k in keys]
        ax.bar(np.arange(len(keys)) + i * width, vals, width, color=colour,
               label=label)
    for j, k in enumerate(keys):
        n = max((len(v) for v in cells[k].values()), default=0)
        ax.text(j + 0.4, 1.04, f"n={n}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(keys)) + 0.4 - width / 2)
    ax.set_xticklabels([f"{t}\n{l}/{s}" for t, l, s in keys], fontsize=7,
                       rotation=45, ha="right")
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("greedy accuracy")
    ax.set_title(f"capability gate — {title}")
    ax.legend(fontsize=8, ncol=3, loc="upper left",
              bbox_to_anchor=(0, -0.28), frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout()
    save(fig, path)


def lens_overlay(packed, task, ro, path):
    """Same role curve under several lenses, or from two runs."""
    keys = sorted({k for p in packed for k in p["by"] if k[0] == task})
    if not keys:
        return
    scripts = sorted({k[2] for k in keys})
    langs = sorted({k[1] for k in keys})
    fig, ax = plt.subplots(len(scripts), len(langs), squeeze=False,
                           figsize=(3.9 * len(langs), 3.3 * len(scripts)),
                           sharex=True, sharey=True)
    styles = ["-", "--", ":", "-."]
    drew = False
    for i, sc in enumerate(scripts):
        for j, lg in enumerate(langs):
            a = ax[i][j]
            any_here = False
            for p, ls in zip(packed, styles):
                cell = p["by"].get((task, lg, sc))
                if not cell or ro not in cell:
                    continue
                m, ci, n = mean_ci(cell[ro])
                x = np.arange(len(m))
                a.plot(x, m, ls, lw=1.7, label=f"{p['label']} (n={n})")
                a.fill_between(x, m - ci, m + ci, alpha=0.12)
                any_here = True
            if not any_here:
                a.set_visible(False)
                continue
            drew = True
            a.set_title(f"{lg} / {sc}", fontsize=10)
            a.grid(alpha=0.25, lw=0.5)
            if i == len(scripts) - 1:
                a.set_xlabel("layer")
            if j == 0:
                a.set_ylabel(f"P({ro})")
    if not drew:
        plt.close(fig)
        return
    ax[0][0].legend(fontsize=8)
    fig.suptitle(f"P({ro}) — {task}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, path)


def report(run_dir, lenses=None, compare_run=None):
    figs = os.path.join(run_dir, "figures")
    os.makedirs(figs, exist_ok=True)
    lenses = lenses or available_lenses(run_dir)
    base = load_run(run_dir, lenses[0])
    name = base["name"]
    print(f"[report] {name}  lenses={lenses}")

    accuracy_figure(base["records"], f"{figs}/accuracy.png", name)
    tasks = sorted({r["task"] for r in base["records"]})

    packed = []
    for lens in lenses:
        d = load_run(run_dir, lens)
        d["by"] = collect(d["records"], d["curves"])
        d["by_rank"] = collect(d["records"], d["ranks"])
        d["label"] = lens
        packed.append(d)
        for task in tasks:
            curves_figure(d["by"], task,
                          f"P(language) — {task} — {lens} — {name}",
                          f"{figs}/curves__{task}__{lens}.png")
            curves_figure(d["by_rank"], task,
                          f"rank of best token — {task} — {lens}",
                          f"{figs}/ranks__{task}__{lens}.png",
                          logy=True, offset=1)
            by_script = collect_scripts(d["records"], d["curves"], task)
            scripts_figure(by_script, task,
                           f"probability mass by script — {task}",
                           f"{figs}/scripts__{task}.png")
            switch_figure(d["by_rank"], by_script, task,
                          f"{figs}/switch__{task}__{lens}.png",
                          n_layers=d["config"].get("n_layers"))

    if len(packed) > 1:
        for task in tasks:
            for ro in ("en", "native", "roman"):
                lens_overlay(packed, task, ro,
                             f"{figs}/lenscmp__{task}__{ro}.png")

    if compare_run:
        other = load_run(compare_run, lenses[0])
        other["by"] = collect(other["records"], other["curves"])
        other["label"] = other["name"]
        me = {**packed[0], "label": name}
        cmp_dir = os.path.join(figs, "compare")
        os.makedirs(cmp_dir, exist_ok=True)
        for task in tasks:
            for ro in ("en", "native", "roman"):
                lens_overlay([me, other], task, ro,
                             f"{cmp_dir}/modelcmp__{task}__{ro}.png")

    rows = summary_rows(run_dir, lenses)
    table = render_table(rows)
    open(f"{run_dir}/summary.md", "w").write(table + "\n")
    print("\n" + table)
    print(f"\n[report] figures in {figs}")
    return rows