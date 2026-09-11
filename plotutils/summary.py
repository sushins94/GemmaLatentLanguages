"""Aggregation and the summary table.

Rank statistics are the ones to compare ACROSS lenses. A Jacobian transport
rescales the residual, which changes the effective softmax temperature, so
P_en under J-lens is not on the same scale as under logit lens. Ranks are
scale-invariant, which is why the Qwen replication used harmonic-mean-of-rank
(= 1/MRR).
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

from utils.vocab import LANG_SCRIPT, role_of

COLS = ["run", "lens", "task", "lang", "script", "shots", "n",
        "acc", "acc_exact", "acc_tok1",
        "L_en", "P_en", "P_enctrl", "en_ratio", "L_tgt", "P_tgt",
        "L_switch", "sw_ref", "sw_med", "sw_sd", "sw_never",
        "MRR_en", "MRR_tgt"]


def available_lenses(run_dir):
    return sorted(os.path.basename(p)[len("curves__"):-len(".npz")]
                  for p in glob.glob(f"{run_dir}/curves__*.npz"))


def load_run(run_dir, lens="logit"):
    d = {"name": os.path.basename(os.path.normpath(run_dir)), "dir": run_dir,
         "lens": lens}
    d["records"] = json.load(open(f"{run_dir}/records.json"))
    d["config"] = json.load(open(f"{run_dir}/config.json"))
    d["curves"] = dict(np.load(f"{run_dir}/curves__{lens}.npz"))
    d["ranks"] = dict(np.load(f"{run_dir}/ranks__{lens}.npz"))
    return d


def collect(records, data, roles=True):
    """-> {(task, language, script): {role: [arrays]}}"""
    index = defaultdict(list)
    for key, arr in data.items():
        rid, _, label = key.partition("::")
        index[rid].append((role_of(label) if roles else label, arr))
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        if not r.get("kept", True):
            continue
        cell = (r["task"], r["language"], r["script"])
        for ro, arr in index.get(r["id"], []):
            if ro and ro != "other":
                by[cell][ro].append(arr)
    return by


def collect_scripts(records, curves, task=None):
    """-> {(task, language, script): {SCRIPT_NAME: [arrays]}}"""
    index = defaultdict(list)
    for key, arr in curves.items():
        rid, _, label = key.partition("::")
        if label.startswith("script_"):
            index[rid].append((label[len("script_"):], arr))
    by = defaultdict(lambda: defaultdict(list))
    for r in records:
        if (task and r["task"] != task) or not r.get("kept", True):
            continue
        for scr, arr in index.get(r["id"], []):
            by[(r["task"], r["language"], r["script"])][scr].append(arr)
    return by


def mean_ci(arrs):
    a = np.stack(arrs)
    m = a.mean(0)
    ci = (1.96 * a.std(0, ddof=1) / np.sqrt(len(a)) if len(a) > 1
          else np.zeros_like(m))
    return m, ci, len(a)


def per_prompt_switch(ranks_cell, target_role):
    """Crossover layer for EACH prompt, not the crossover of the mean curves.

    If half the prompts switch at 22 and half at 30, the mean curve crosses
    near 26 -- a layer at which no individual prompt switched. Returns -1 where
    the target never overtakes English.
    """
    if "en" not in ranks_cell or target_role not in ranks_cell:
        return np.array([])
    en, tg = np.stack(ranks_cell["en"]), np.stack(ranks_cell[target_role])
    out = []
    for a, b in zip(en, tg):
        w = np.where(b < a)[0]
        out.append(int(w[0]) if len(w) else -1)
    return np.array(out)


def per_prompt_script_switch(script_cell, language):
    """Layer where target-SCRIPT mass first exceeds LATIN mass, per prompt.

    The fallback for classes with no English answer, where rank against
    en_control is degenerate (an arbitrary English word is always weak, so the
    target 'overtakes' it at layer 0). Needs no answer key, so it is also the
    one switch measure comparable across every class.
    """
    tgt = LANG_SCRIPT.get(language)
    if not tgt or tgt not in script_cell or "LATIN" not in script_cell:
        return np.array([])
    lat, nat = np.stack(script_cell["LATIN"]), np.stack(script_cell[tgt])
    out = []
    for a, b in zip(lat, nat):
        w = np.where(b > a)[0]
        out.append(int(w[0]) if len(w) else -1)
    return np.array(out)


def _fill_switch(s, sw):
    good = sw[sw >= 0]
    s["sw_never"] = int((sw < 0).sum())
    if len(good):
        s["sw_med"] = float(np.median(good))
        s["sw_sd"] = float(good.std())


def cell_stats(curves_cell, ranks_cell, target_role, script_cell=None,
               language=None):
    s = {}
    if "en" in curves_cell:
        m = np.stack(curves_cell["en"]).mean(0)
        s["L_en"], s["P_en"] = int(m.argmax()), float(m.max())
    if target_role in curves_cell:
        m = np.stack(curves_cell[target_role]).mean(0)
        s["L_tgt"], s["P_tgt"] = int(m.argmax()), float(m.max())
    if "en_ctrl" in curves_cell:
        m = np.stack(curves_cell["en_ctrl"]).mean(0)
        s["P_enctrl"] = float(m.max())
        if "P_en" in s:
            s["en_ratio"] = float(s["P_en"] / max(m.max(), 1e-9))
    for ro, key in (("en", "MRR_en"), (target_role, "MRR_tgt")):
        if ro in ranks_cell:
            r = np.stack(ranks_cell[ro]).astype(float) + 1.0
            s[key] = float((1.0 / r).mean())

    if "en" in ranks_cell and target_role in ranks_cell:
        s["sw_ref"] = "en"
        a = np.stack(ranks_cell["en"]).mean(0)
        b = np.stack(ranks_cell[target_role]).mean(0)
        w = np.where(b < a)[0]
        s["L_switch"] = int(w[0]) if len(w) else -1
        _fill_switch(s, per_prompt_switch(ranks_cell, target_role))
    elif script_cell and language:
        sw = per_prompt_script_switch(script_cell, language)
        if len(sw):
            s["sw_ref"] = "script"
            m = np.stack(script_cell["LATIN"]).mean(0)
            n = np.stack(script_cell[LANG_SCRIPT[language]]).mean(0)
            w = np.where(n > m)[0]
            s["L_switch"] = int(w[0]) if len(w) else -1
            _fill_switch(s, sw)
    return s


def summary_rows(run_dir, lenses=None):
    rows = []
    for lens in (lenses or available_lenses(run_dir)):
        d = load_run(run_dir, lens)
        acc = defaultdict(lambda: defaultdict(list))
        shots = {}
        for r in d["records"]:
            if not r.get("kept", True):
                continue
            key = (r["task"], r["language"], r["script"])
            shots[key] = r.get("n_demos", "")
            for m in ("correct", "correct_exact", "correct_first_token"):
                if r.get(m) is not None:
                    acc[key][m].append(bool(r[m]))
        cur = collect(d["records"], d["curves"])
        rnk = collect(d["records"], d["ranks"])
        scr = collect_scripts(d["records"], d["curves"])
        for cell in sorted(cur):
            task, lang, script = cell
            row = {"run": d["name"], "lens": lens, "task": task, "lang": lang,
                   "script": script, "shots": shots.get(cell, ""),
                   "n": max((len(v) for v in cur[cell].values()), default=0)}
            for m, col in (("correct", "acc"), ("correct_exact", "acc_exact"),
                           ("correct_first_token", "acc_tok1")):
                v = acc[cell].get(m)
                row[col] = float(np.mean(v)) if v else np.nan
            row.update(cell_stats(cur[cell], rnk.get(cell, {}),
                                  "native" if script == "native" else "roman",
                                  script_cell=scr.get(cell), language=lang))
            rows.append(row)
    return rows


def render_table(rows):
    cols = [c for c in COLS if any(c in r for r in rows)]
    def f(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}".rstrip("0").rstrip(".")
        return str(v)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(r.get(c, "")) for c in cols) + " |"
              for r in rows]
    return "\n".join(lines)