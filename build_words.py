#!/usr/bin/env python
"""
Build and validate the word list.

    # 1. get a blank sheet to fill, or to see the expected columns
    python build_words.py --template promptutils/words_template.csv

    # 2. validate a filled sheet against the tokenizer and write words.json
    python build_words.py --input data/my_words.csv \
        --model ../data/models/gemma-3-4b-pt/

    # 3. inspect without rewriting anything
    python build_words.py --check --model ../data/models/gemma-3-4b-pt/

Step 2 writes `data/words.json`, which promptgen/words.py picks up
automatically. Nothing else changes; `run.py sets` regenerates every class from
the new list.

POTENTIAL SOURCES 
------------------------------------------
  google-research-datasets/dakshina
        Romanisation lexicons for 12 South Asian languages including hi/ta/ml:
        native word -> attested romanisations, produced by native speakers.
        This is the cheapest way to fill the *_roman columns, and better than
        transliterating yourself.

  AI4Bharat/Romanlens
        Indic words already filtered for exactly this task, with IndicXlit
        romanisations.

  epfl-dlab/llm-latent-language
        Wendler's own lists (en/de/fr/ru/zh). Adding the zh columns turns
        class0 from a one-prompt gate into a full Chinese replication you can
        compare your Indic numbers against.

  facebookresearch/MUSE
        Bilingual en->xx dictionaries, useful for the translation columns.

MERGING
-------
Pass several files; later ones fill gaps in earlier ones, matched on `en`:

    python build_words.py --input base.csv dakshina_roman.csv muse_ta.csv \
        --model ../data/models/gemma-3-4b-pt/

So you can keep one file per source and never hand-merge. Rows are only
complete when every {hi,ta,ml}_native column is filled; incomplete rows are
reported and skipped.


VALIDATION
------------
Wendler's measurement needs the first token of the answer to identify the
language unambiguously. Two things break that, and both are language-dependent:

  fertility   how many tokens the word costs. Gemma 3's 262k tokenizer covers
              Indic scripts far better than Llama-2's did, but coverage still
              varies, and a word that costs 6 tokens carries little signal in
              its first one.
  collision   if English and the target share a starting token, P(en) and
              P(target) cannot be told apart and the item is dropped. Latin
              script is shared with English, so romanised conditions are hit
              hardest.

The report below tells you the surviving n per language BEFORE spending GPU
time.
"""

import argparse
import csv
import json
import re
import os
import sys

LANGS = ("hi", "ta", "ml")
COLUMNS = (["en", "fr", "de", "cognate"]
           + [f"{lg}_{k}" for lg in LANGS for k in ("native", "roman")]
           + ["cloze_en"] + [f"cloze_{lg}" for lg in LANGS])


def write_template(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerow(["flower", "fleur", "Blume", "0",
                    "फूल", "phool", "பூ", "poo", "പൂവ്", "poovu",
                    "The bee landed on the ___",
                    "मधुमक्खी ___ पर बैठी",
                    "தேனீ ___ மீது அமர்ந்தது",
                    "തേനീച്ച ___ മേൽ ഇരുന്നു"])
    print(f"[template] {path}")
    print(f"[template] columns: {', '.join(COLUMNS)}")
    print("[template] cognate = 1 when the Indic word is an English loanword "
          "(road, bus, doctor). That flag is the control for the objection "
          "that English and target unembedding rows are near-duplicates -- "
          "keep it honest.")


# ---------------------------------------------------------------------------
# RomanLens importer
# ---------------------------------------------------------------------------

RL_LANG_DIR = {"en": "en", "fr": "fr", "de": "de", "zh": "zh",
               "hi": "hi", "ta": "ta", "ml": "ml",
               "hi_roman": "hi_translit", "ta_roman": "ta_translit",
               "ml_roman": "ml_translit"}


def _unquote(s):
    """RomanLens cloze frames are double-escaped by CSV round-tripping, so a
    quote appears as """" or "". Collapse runs of quotes to one."""
    return re.sub(r'"{2,}', '"', s).strip()


def _rl_read(root, sub, col="word_translation", fname="clean6.csv"):
    """Read one RomanLens language file -> {english_word: primary_form}.

    Files are aligned on `word_original`, the English word, which is what makes
    this a one-shot import: every language table keys off the same 132 words.

    `word_translation` holds comma-separated synonyms; the first is the primary
    form and the rest are alternatives their filtering kept. We take the first.

    NOTE: the `lang` column is unreliable -- several Indic files carry "ml"
    regardless of their actual content -- so the directory name is used instead.
    """
    path = os.path.join(root, sub, fname)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("word_original") or "").strip().lower()
            val = (r.get(col) or "").strip()
            if not (key and val):
                continue
            out[key] = (_unquote(val) if col != "word_translation"
                        else val.split(",")[0].strip())
    return out


def _sim(a, b):
    """Normalised similarity between two Latin strings, 0 to 1."""
    from difflib import SequenceMatcher
    a, b = a.lower().strip(), b.lower().strip()
    return SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def flag_cognates(rows, langs=LANGS, threshold=0.6):
    """Mark items whose Indic form is orthographically close to the English.

    RomanLens ships no cognate column, so without this every row imports as
    cognate=0 and the control is empty.

    A cognate here means the target word IS the English word borrowed, so its
    romanisation looks like the English spelling: bag/bagg, doctor/doctor,
    bus/bas. Comparing the ENGLISH word against the ROMANISATION is the right
    test, since both are Latin script and the native form is not comparable.

    This matters because it is the objection about near-duplicate unembedding
    rows made testable: if the mid-layer English signal is the same for
    cognates and non-cognates, lexical similarity is not driving it. An
    orthographic proxy is imperfect (it will miss semantic borrowings that were
    respelled, and catch coincidences), so treat the split as a coarse one and
    report the threshold.
    """
    n, hits = 0, []
    for r in rows:
        scored = [(_sim(r["en"], r.get(f"{lg}_roman", "")), lg)
                  for lg in langs]
        best, lg = max(scored)
        r["cognate"] = "1" if best >= threshold else "0"
        r["cognate_sim"] = f"{best:.2f}"
        r["cognate_lang"] = lg if best >= threshold else ""
        if best >= threshold:
            n += 1
            hits.append((best, r["en"], lg, r.get(f"{lg}_roman", "")))
    print(f"[cognate] flagged {n}/{len(rows)} at similarity >= {threshold}")
    for s_, e, lg, ro in sorted(hits, reverse=True)[:8]:
        print(f"[cognate]   {e} / {lg}:{ro}  ({s_:.2f})")
    if n < 10:
        print("[cognate] too few to split on. Report the control as "
              "underpowered, or hand-flag loanwords in data/words.json.")
    return rows


def load_romanlens(root, langs=LANGS):
    """Build entries from a RomanLens checkout.

    root points at llm_logit_lens/data/langs inside
    github.com/AI4Bharat/Romanlens.

    Supplies EVERY column this project needs in one step: English, French and
    German source words, native-script targets, romanisations from the
    *_translit tables, cloze frames from clean_cloze.csv, and Chinese -- which
    turns class0 from a one-prompt gate into a full replication of Wendler's
    original to compare your Indic numbers against.
    """
    tables = {k: _rl_read(root, d) for k, d in RL_LANG_DIR.items()}
    cloze_en = _rl_read(root, "en", col="blank_prompt_translation_masked")
    cloze = {lg: _rl_read(root, lg, col="blank_prompt_translation_masked",
                          fname="clean_cloze.csv") for lg in langs}

    for k, t in tables.items():
        print(f"[romanlens] {k:<10} {len(t)} words")

    keys = [k for k in tables["en"] if all(tables[lg].get(k) for lg in langs)
            and all(tables[f"{lg}_roman"].get(k) for lg in langs)]
    print(f"[romanlens] {len(keys)} words complete across "
          f"{'/'.join(langs)} native + roman")

    rows = []
    for k in keys:
        r = {"en": tables["en"][k],
             "fr": tables["fr"].get(k, ""),
             "de": tables["de"].get(k, ""),
             "zh": tables["zh"].get(k, ""),
             "cognate": "0", "cloze_en": cloze_en.get(k, "")}
        for lg in langs:
            r[f"{lg}_native"] = tables[lg][k]
            r[f"{lg}_roman"] = tables[f"{lg}_roman"][k]
            r[f"cloze_{lg}"] = cloze[lg].get(k, "")
        rows.append(r)

    n_fr = sum(bool(r["fr"]) for r in rows)
    n_cl = sum(all(r.get(f"cloze_{lg}") for lg in langs) for r in rows)
    print(f"[romanlens] {n_fr}/{len(rows)} have French (needed for the "
          f"fr-source condition); {n_cl} have cloze frames in all languages")
    return rows


def merge_rows(files):
    """Merge several sheets on `en`; later files fill gaps, never overwrite.

    Lets you keep one file per source -- translations from one, romanisations
    from another -- instead of hand-merging into a single sheet.
    """
    merged, order = {}, []
    for path in files:
        rows = read_rows(path)
        added = filled = 0
        for r in rows:
            key = (r.get("en") or "").strip().lower()
            if not key:
                continue
            if key not in merged:
                merged[key] = dict(r)
                order.append(key)
                added += 1
                continue
            for k, v in r.items():
                if v and not (merged[key].get(k) or "").strip():
                    merged[key][k] = v
                    filled += 1
        print(f"[merge] {path}: {len(rows)} rows -> "
              f"{added} new, {filled} gaps filled")
    return [merged[k] for k in order]


def read_rows(path):
    if path.endswith(".json"):
        return json.load(open(path, encoding="utf-8"))
    delim = "\t" if path.endswith((".tsv", ".tab")) else ","
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def to_entry(row):
    e = {"en": (row.get("en") or "").strip(),
         "fr": (row.get("fr") or "").strip(),
         "de": (row.get("de") or "").strip(),
         "zh": (row.get("zh") or "").strip(),
         "cognate": str(row.get("cognate", "0")).strip() in ("1", "true",
                                                             "True", "yes")}
    for lg in LANGS:
        nat = (row.get(f"{lg}_native") or "").strip()
        rom = (row.get(f"{lg}_roman") or "").strip()
        if not nat:
            return None, f"missing {lg}_native"
        e[lg] = (nat, rom or nat)
    cloze = {k: (row.get(f"cloze_{k}") or "").strip()
             for k in ("en",) + LANGS}
    if all(cloze.values()):
        e["cloze"] = cloze
    if not e["en"]:
        return None, "missing en"
    return e, None


def validate(entries, model_path):
    """Report fertility, single-token counts and collision survival."""
    from transformers import AutoTokenizer
    from utils.vocab import VocabIndex, build_token_sets, has_collision

    tok = AutoTokenizer.from_pretrained(model_path)
    index = VocabIndex(tok)
    print(f"\n[vocab] tokenizer {len(tok)} tokens\n")

    print(f"{'lang':>5} {'cond':>7} {'n':>4} {'1tok':>5} {'fert':>6} "
          f"{'kept':>5} {'kept%':>6}")
    summary = {}
    for lg in LANGS:
        for cond in ("native", "roman"):
            ferts, single, kept = [], 0, 0
            for e in entries:
                word = e[lg][0 if cond == "native" else 1]
                ferts.append(index.fertility(word))
                single += index.is_single_token(word)
                ts = build_token_sets(index, {
                    "en": e["en"], f"{lg}_{cond}": word})
                kept += not has_collision(ts, ["en", f"{lg}_{cond}"])
            n = len(entries)
            summary[f"{lg}_{cond}"] = kept
            print(f"{lg:>5} {cond:>7} {n:>4} {single:>5} "
                  f"{sum(ferts) / max(n, 1):>6.2f} {kept:>5} "
                  f"{100 * kept / max(n, 1):>5.0f}%")

    n_cloze = sum("cloze" in e for e in entries)
    n_cog = sum(e["cognate"] for e in entries)
    print(f"\n[cloze]   {n_cloze}/{len(entries)} rows have frames "
          f"(class2 uses only these)")
    print(f"[cognate] {n_cog}/{len(entries)} flagged as English loanwords")

    worst = min(summary, key=summary.get)
    if summary[worst] < 30:
        print(f"\n[warn] {worst} keeps only {summary[worst]} items after the "
              f"collision filter. Aim for 60-100 per condition; below ~30 the "
              f"confidence intervals will swamp any effect you are looking "
              f"for.")
    return summary


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--template", default=None, help="write a blank CSV here")
    ap.add_argument("--input", default=None, nargs="+",
                    help="one or more CSV/TSV/JSON sheets; later files fill "
                         "gaps in earlier ones, matched on `en`")
    ap.add_argument("--out", default="promptutils/words.json")
    ap.add_argument("--model", default=None,
                    help="tokenizer to validate against")
    ap.add_argument("--cognate-threshold", type=float, default=0.6,
                    help="orthographic similarity between the English word "
                         "and its romanisation above which an item is flagged "
                         "as a loanword")
    ap.add_argument("--romanlens", default=None,
                    help="path to Romanlens/llm_logit_lens/data/langs; "
                         "supplies en/fr/de/zh, native scripts, "
                         "romanisations and cloze frames in one step")
    ap.add_argument("--check", action="store_true",
                    help="validate the CURRENT word list, write nothing")
    a = ap.parse_args()

    if a.template:
        write_template(a.template)
        return

    if a.check:
        from promptutils.words import IS_PLACEHOLDER, WORDS
        print(f"[check] {len(WORDS)} entries "
              f"({'PLACEHOLDER' if IS_PLACEHOLDER else 'data/words.json'})")
        if not a.model:
            raise SystemExit("--check needs --model")
        validate(WORDS, a.model)
        return

    if not a.input and not a.romanlens:
        raise SystemExit("pass --romanlens, --input, --template or --check")

    if a.romanlens:
        rows = load_romanlens(a.romanlens)
        rows = flag_cognates(rows, threshold=a.cognate_threshold)
        if a.input:
            rows = merge_rows(a.input) + rows      # your sheets take priority
    else:
        rows = (merge_rows(a.input) if len(a.input) > 1
                else read_rows(a.input[0]))
    entries, bad = [], []
    for i, r in enumerate(rows, 2):        # row 1 is the header
        e, err = to_entry(r)
        (entries.append(e) if e else bad.append((i, err)))
    print(f"[read] {len(entries)} usable, {len(bad)} skipped")
    for i, err in bad[:10]:
        print(f"   line {i}: {err}")

    if a.model:
        validate(entries, a.model)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump([{**e, **{lg: list(e[lg]) for lg in LANGS}}
                   for e in entries], f, ensure_ascii=False, indent=2)
    print(f"\n[wrote] {a.out} ({len(entries)} entries)")
    print("[next]  python run.py sets --out sets/")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()