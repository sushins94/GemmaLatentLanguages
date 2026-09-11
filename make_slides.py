#!/usr/bin/env python3
"""
Build a Beamer PDF per word list from the outputs tree.

    python3 make_slides.py                      # everything under outputs/
    python3 make_slides.py --layout wordlist    # one combined deck per list
    python3 make_slides.py --no-compile         # write .tex only

Expects the layout run.py produces:

    outputs/<wordlist>/<model>/<promptset>/
        config.json
        summary.md
        figures/*.png
        figures/compare/*.png

Default output ("model" layout), one small deck per model per prompt class:

    slides/<wordlist>/<promptset>/<model>.pdf

Word lists never mix, each prompt class gets its own folder, and each model has
its own PDF so two models can be opened side by side. With --layout wordlist
you instead get one combined deck per word list, sectioned by model.

Either way: the summary table for the class, then one figure per slide. Frame
titles name the observable, and the caption is a link to the PNG on disk, so a
slide can be traced back to what produced it.

Purely a directory scan with no cached state, so it is safe to rerun whenever
more runs finish. It overwrites existing PDFs.

Compiles with pdflatex if available, twice so cross-references resolve.
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess

# figure filename -> human-readable observable
PATTERNS = [
    (r"^accuracy$",
     "Capability gate: greedy accuracy (substring / exact / first token)"),
    (r"^prefix_sweep$",
     "Prefix sweep: accuracy against number of prefix words given"),
    (r"^curves__(?P<task>.+)__(?P<lens>[^_]+)$",
     "P(language) per layer, {task} ({lens} lens)"),
    (r"^ranks__(?P<task>.+)__(?P<lens>[^_]+)$",
     "Rank of the best token per layer, {task} ({lens} lens)"),
    (r"^scripts__(?P<task>.+)$",
     "Probability mass by Unicode script per layer, {task}"),
    (r"^switch__(?P<task>.+)__(?P<lens>[^_]+)$",
     "Per-prompt layer at which the target overtakes English, {task}"),
    (r"^lenscmp__(?P<task>.+)__(?P<role>[^_]+)$",
     "Lens comparison: P({role}) per layer, {task}"),
    (r"^modelcmp__(?P<task>.+)__(?P<role>[^_]+)$",
     "Model comparison: P({role}) per layer, {task}"),
    (r"^compare__(?P<role>[^_]+)__task-(?P<task>.+)$",
     "Model comparison: P({role}) per layer, {task}"),
]

# summary.md is wide; these are the columns worth putting on a slide
KEEP_COLS = ["lens", "task", "lang", "script", "shots", "n",
             "acc", "acc_tok1", "L_en", "P_en", "P_enctrl", "en_ratio",
             "L_tgt", "P_tgt", "L_switch", "sw_ref", "sw_med", "sw_sd"]
ROWS_PER_SLIDE = 12


def esc(s):
    """Escape LaTeX specials. Figure names and summary cells are ASCII."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def observable(stem):
    for pat, label in PATTERNS:
        m = re.match(pat, stem)
        if m:
            return label.format(**{k: v.replace("_", " ")
                                   for k, v in m.groupdict().items()})
    return stem.replace("_", " ")


def read_summary(path):
    """Parse the markdown table in summary.md -> (header, rows)."""
    if not os.path.exists(path):
        return None, []
    lines = [l.strip() for l in open(path) if l.strip().startswith("|")]
    if len(lines) < 3:
        return None, []
    def cells(l):
        return [c.strip() for c in l.strip("|").split("|")]
    header = cells(lines[0])
    rows = [cells(l) for l in lines[2:]]
    keep = [i for i, h in enumerate(header) if h in KEEP_COLS]
    if not keep:
        keep = list(range(min(len(header), 10)))
    return ([header[i] for i in keep],
            [[r[i] if i < len(r) else "" for i in keep] for r in rows])


def table_slides(title, header, rows, out):
    if not header or not rows:
        return
    for start in range(0, len(rows), ROWS_PER_SLIDE):
        chunk = rows[start:start + ROWS_PER_SLIDE]
        part = ("" if len(rows) <= ROWS_PER_SLIDE
                else f" ({start // ROWS_PER_SLIDE + 1})")
        out.append(r"\begin{frame}{" + esc(title) + part + "}")
        # resizebox scales the table to the slide width, so a 16-column
        # summary stays legible instead of shrinking to nothing
        out.append(r"\vfill\begin{center}")
        out.append(r"\resizebox{\textwidth}{!}{%")
        out.append(r"\begin{tabular}{" + "l" * len(header) + "}")
        out.append(r"\hline")
        out.append(" & ".join(r"\textbf{" + esc(h) + "}" for h in header)
                   + r" \\ \hline")
        for r in chunk:
            out.append(" & ".join(esc(c) for c in r) + r" \\")
        out.append(r"\hline\end{tabular}}")
        out.append(r"\end{center}\vfill")
        out.append(r"\end{frame}")


def figure_slide(png, root_abs, out, slides_dir):
    """One figure per slide. The caption is a link to the PNG on disk.

    The link target is relative to the slides directory rather than absolute,
    so moving or copying the whole project keeps it working. The visible text
    is the path relative to the outputs root, which is what identifies the run.
    """
    stem = os.path.splitext(os.path.basename(png))[0]
    title = observable(stem)
    rel = os.path.relpath(os.path.abspath(png), root_abs)
    link = os.path.relpath(os.path.abspath(png), os.path.abspath(slides_dir))
    out.append(r"\begin{frame}{" + esc(title) + "}")
    out.append(r"\begin{center}")
    out.append(r"\includegraphics[width=\textwidth,height=0.78\textheight,"
               r"keepaspectratio]{" + os.path.abspath(png) + "}")
    out.append(r"\end{center}")
    out.append(r"\vspace{-2mm}{\tiny\href{run:" + link + r"}{\texttt{"
               + esc(rel) + r"}}}")
    out.append(r"\end{frame}")


def sort_key(png):
    """Order figures so a class reads: gate, curves, scripts, ranks, switch,
    then comparisons."""
    order = ["accuracy", "prefix_sweep", "curves__", "scripts__", "ranks__",
             "switch__", "lenscmp__", "modelcmp__", "compare__"]
    b = os.path.basename(png)
    for i, p in enumerate(order):
        if b.startswith(p):
            return (i, b)
    return (len(order), b)


def preamble(title, subtitle):
    return [
        r"\documentclass[aspectratio=169,10pt]{beamer}",
        r"\usepackage{graphicx}",
        # beamer loads hyperref itself; passing options here causes an option
        # clash, so set them through hypersetup after the class instead
        r"\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=.}",
        r"\usetheme{default}\setbeamertemplate{navigation symbols}{}",
        r"\setbeamerfont{frametitle}{size=\small}",
        r"\title{" + esc(title) + "}",
        r"\subtitle{" + esc(subtitle) + "}",
        r"\date{\today}",
        r"\begin{document}",
        r"\frame{\titlepage}",
    ]


def config_slide(sdir, heading, out):
    p = os.path.join(sdir, "config.json")
    if not os.path.exists(p):
        return
    try:
        cfg = json.load(open(p))
    except Exception:                                    # noqa: BLE001
        return
    out.append(r"\begin{frame}{" + esc(heading) + r"}\small")
    out.append(r"\begin{itemize}")
    for k in ("n_layers", "d_model", "vocab_head", "vocab_tokenizer",
              "weight_dtype", "softcap", "n_prompts", "n_collisions",
              "placeholder_words", "n_words"):
        if k in cfg:
            out.append(r"\item " + esc(f"{k}: {cfg[k]}"))
    out.append(r"\end{itemize}\end{frame}")


def figures_in(sdir):
    return sorted(
        glob.glob(os.path.join(sdir, "figures", "*.png"))
        + glob.glob(os.path.join(sdir, "figures", "compare", "*.png")),
        key=sort_key)


def build_one(sdir, model, pset, wl, out_tex, root_abs, slides_dir):
    """One deck for a single (model, prompt class)."""
    figs = figures_in(sdir)
    if not figs:
        return 0
    out = preamble(f"{model}", f"{wl} / {pset}")
    config_slide(sdir, f"{model} / {pset}", out)
    header, rows = read_summary(os.path.join(sdir, "summary.md"))
    table_slides(f"{model} / {pset}: summary", header, rows, out)
    for png in figs:
        figure_slide(png, root_abs, out, slides_dir)
    out.append(r"\end{document}")
    os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)
    open(out_tex, "w").write("\n".join(out) + "\n")
    return len(figs)


def build_wordlist(wordlist_dir, out_tex, root_abs, slides_dir):
    """One combined deck for a whole word list, sectioned by model."""
    wl = os.path.basename(os.path.normpath(wordlist_dir))
    models = sorted(d for d in glob.glob(os.path.join(wordlist_dir, "*"))
                    if os.path.isdir(d))
    out = preamble("Latent language of Gemma 3", f"Word list: {wl}")
    out.append(r"\begin{frame}{Contents}\footnotesize"
               r"\tableofcontents\end{frame}")
    n_fig = 0
    for mdir in models:
        model = os.path.basename(mdir)
        sets = sorted(d for d in glob.glob(os.path.join(mdir, "*"))
                      if os.path.isdir(d))
        if not sets:
            continue
        out.append(r"\section{" + esc(model) + "}")
        config_slide(sets[0], model, out)
        for sdir in sets:
            pset = os.path.basename(sdir)
            out.append(r"\subsection{" + esc(pset) + "}")
            header, rows = read_summary(os.path.join(sdir, "summary.md"))
            table_slides(f"{model} / {pset}: summary", header, rows, out)
            for png in figures_in(sdir):
                figure_slide(png, root_abs, out, slides_dir)
                n_fig += 1
    out.append(r"\end{document}")
    os.makedirs(os.path.dirname(out_tex) or ".", exist_ok=True)
    open(out_tex, "w").write("\n".join(out) + "\n")
    return n_fig, len(models)


def compile_pdf(tex, engine="pdflatex"):
    if not shutil.which(engine):
        print(f"[skip] {engine} not on PATH; .tex written but not compiled")
        return False
    d = os.path.dirname(os.path.abspath(tex)) or "."
    for i in range(2):          # twice, so the table of contents resolves
        r = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error",
             os.path.basename(tex)],
            cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[error] {engine} failed on pass {i+1}")
            tail = [l for l in r.stdout.splitlines() if l.startswith("!")][:5]
            for l in tail:
                print("   " + l)
            return False
    for ext in (".aux", ".log", ".nav", ".out", ".snm", ".toc"):
        p = tex[:-4] + ext
        if os.path.exists(p):
            os.remove(p)
    return True


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs")
    ap.add_argument("--out", default="slides")
    ap.add_argument("--layout", default="model", choices=["model", "wordlist"],
                    help="'model': slides/<wordlist>/<promptset>/<model>.pdf. "
                         "'wordlist': one combined deck per word list.")
    ap.add_argument("--models", nargs="*", default=None,
                    help="only these model directories")
    ap.add_argument("--sets", nargs="*", default=None,
                    help="only these prompt-set directories")
    ap.add_argument("--engine", default="pdflatex")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    root_abs = os.path.abspath(a.root)
    wordlists = sorted(d for d in glob.glob(os.path.join(a.root, "*"))
                       if os.path.isdir(d))
    if not wordlists:
        raise SystemExit(f"nothing under {a.root}/")
    os.makedirs(a.out, exist_ok=True)

    if a.layout == "wordlist":
        for wdir in wordlists:
            wl = os.path.basename(os.path.normpath(wdir))
            tex = os.path.join(a.out, f"{wl}.tex")
            n_fig, n_model = build_wordlist(wdir, tex, root_abs, a.out)
            print(f"[{wl}] {n_model} models, {n_fig} figures -> {tex}")
            if not a.no_compile and n_fig and compile_pdf(tex, a.engine):
                print(f"[{wl}] wrote {tex[:-4]}.pdf")
        return

    made = 0
    for wdir in wordlists:
        wl = os.path.basename(os.path.normpath(wdir))
        for mdir in sorted(glob.glob(os.path.join(wdir, "*"))):
            if not os.path.isdir(mdir):
                continue
            model = os.path.basename(mdir)
            if a.models and model not in a.models:
                continue
            for sdir in sorted(glob.glob(os.path.join(mdir, "*"))):
                if not os.path.isdir(sdir):
                    continue
                pset = os.path.basename(sdir)
                if a.sets and pset not in a.sets:
                    continue
                # slides/<wordlist>/<promptset>/<model>.pdf
                d = os.path.join(a.out, wl, pset)
                tex = os.path.join(d, f"{model}.tex")
                n_fig = build_one(sdir, model, pset, wl, tex, root_abs, d)
                if not n_fig:
                    continue
                print(f"[{wl}/{pset}] {model}: {n_fig} figures")
                if not a.no_compile and compile_pdf(tex, a.engine):
                    made += 1
    print(f"\n{made} PDFs under {a.out}/")


if __name__ == "__main__":
    main()