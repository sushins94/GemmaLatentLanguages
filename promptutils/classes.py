"""All prompt-class builders in one place.

Word lists live in promptgen/words.py. Every class here reads from it.

    class0_wendler      Wendler's own example -- correctness check
    class1_translation  translation, native + romanised
    class2_cloze        cloze completion
    class3_repetition   copy task (syntactic control)
    class4_three_shot   classes 1-3 at 3 demonstrations
    class5_lang_swap    demos in one language, tag in the next
    class6_epic         verse continuation      (needs a text file)
    class7_technical    FLORES completion       (needs `run.py data --flores`)
    class8_gibberish    script-shaped nonsense  (null)

RECORD FORMAT (downstream code reads only the JSON)
{
  "id", "task", "src_lang", "language", "script", "target_label",
  "prompt", "answers": {label: surface form},
  "collision_labels": [labels that must stay mutually distinct]
}

TRACKED LABELS. Each item carries more surface forms than it needs, because
extra labels are free at capture time and each answers an objection:

  en              the English word        Wendler's detour
  <lang>_native   target, native script   the correct answer
  <lang>_roman    target, romanised       RomanLens latent romanisation
  <pivot>_native  a THIRD language        is English special, or would any
                                          high-resource language show up?
  en_control      unrelated English word  does an arbitrary English word peak
                                          just as high? (norm-bias objection)
  src             the source-side word    Wendler reports input-language
                                          probability stays ~0; verify it

Only `collision_labels` must be mutually distinct. Filtering on all six would
destroy the sample, and controls stay readable when they partly overlap.
"""

import os
import random

from .words import EN_CONTROLS, WORDS


def _load_controls():
    """data/controls.json from build_controls.py, if it exists.

    Maps each English word to one matched on unigram frequency and on the size
    of its Start(w) set. Without the file we fall back to EN_CONTROLS, a fixed
    list of common nouns matched on nothing, which gives a coarser check: a low
    control curve is still informative, but a high one cannot distinguish "any
    English word appears here" from "these particular words are frequent".
    """
    import json
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data", "controls.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return {k: v["control"] for k, v in json.load(f).items()}
    return {}


CONTROLS = _load_controls()
CONTROLS_MATCHED = bool(CONTROLS)

# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

TAG = {"en": "English", "fr": "Français", "de": "Deutsch", "zh": "中文",
       "hi": "हिंदी", "ta": "தமிழ்", "ml": "മലയാളം",
       "hi_rom": "Hindi", "ta_rom": "Tamil", "ml_rom": "Malayalam"}

LANGS = ("hi", "ta", "ml")
SWAP = {"ta": "hi", "hi": "ml", "ml": "ta"}      # class 5 cycle
PIVOT = {"hi": "ta", "ta": "hi", "ml": "hi"}     # third language to track


def surface(w, lang):
    if lang in ("en", "fr", "de"):
        return w[lang]
    return w[lang[:-4]][1] if lang.endswith("_rom") else w[lang][0]


def answers_for(w, language, src_lang=None, seed=0):
    rng = random.Random(f"{w['en']}:{language}:{seed}")
    a = {"en": w["en"],
         f"{language}_native": w[language][0],
         f"{language}_roman": w[language][1],
         "en_control": CONTROLS.get(w["en"], rng.choice(EN_CONTROLS))}
    a[f"{PIVOT[language]}_native"] = w[PIVOT[language]][0]
    if src_lang and src_lang != "en" and not src_lang.endswith("_rom"):
        a["src"] = surface(w, src_lang)
    return a


def rec(**kw):
    kw.setdefault("collision_labels", ["en", kw["target_label"]])
    return kw


def demos_for(i, n):
    return [w for j, w in enumerate(WORDS) if j != i][:n]


# --- templates -------------------------------------------------------------

def t_translation(target, demos, src, tgt):
    lines = [f'{TAG[src]}: "{surface(d, src)}" - {TAG[tgt]}: "{surface(d, tgt)}"'
             for d in demos]
    lines.append(f'{TAG[src]}: "{surface(target, src)}" - {TAG[tgt]}: "')
    return "\n".join(lines)


def t_repetition(target, demos, lang):
    lines = [f'{TAG[lang]}: "{surface(d, lang)}" - {TAG[lang]}: "{surface(d, lang)}"'
             for d in demos]
    lines.append(f'{TAG[lang]}: "{surface(target, lang)}" - {TAG[lang]}: "')
    return "\n".join(lines)


def cloze_open(frame):
    """Cut a completed cloze frame back to just before its answer.

    RomanLens frames arrive already answered:

        A "___" is used to read stories. Answer: "book".

    That is exactly right for a demonstration, but the target line must stop at
    the opening quote so the model supplies the word. The answer always sits
    between the last two quotes, so rsplit finds it without needing to know the
    answer label in each language ("Answer", "Réponse", "பதில்", "答案").
    """
    parts = frame.rsplit('"', 2)
    return parts[0] + '"' if len(parts) == 3 else frame.rstrip() + ' "'


def t_cloze(target, demos, lang):
    """Frames live in words.py. In the romanised condition the frame stays in
    native script but the ANSWER is romanised -- how a bilingual speaker types.

    Demonstrations keep the frame's own answer; only the final line is opened.
    """
    base = lang[:-4] if lang.endswith("_rom") else lang
    lines = [d["cloze"][base] for d in demos]
    if lang.endswith("_rom"):
        # the frame's built-in answer is native script; swap in the romanised
        # form so the demonstrated output script matches what we are asking for
        lines = [cloze_open(d["cloze"][base]) + surface(d, lang) + '".'
                 for d in demos]
    lines.append(cloze_open(target["cloze"][base]))
    return "\n".join(lines)


TEMPLATES = {"translation": t_translation, "repetition": t_repetition,
             "cloze": t_cloze}


def usable(words, task, src=None, lang=None):
    """Words that have every surface form this prompt will reference.

    Source-language coverage is incomplete in the RomanLens tables -- French
    has 117 of 132 words, German 120 -- so a translation prompt built from the
    full list yields lines like 'Français: "" - தமிழ்: "'. Cloze needs a frame
    in the target language. Filtering here keeps demonstrations and targets
    drawn from the same usable pool.
    """
    out = []
    for w in words:
        if task == "translation" and src and not (w.get(src) or "").strip():
            continue
        if task == "cloze" and lang and not (w.get("cloze", {}).get(lang)):
            continue
        out.append(w)
    return out


def core(task, n_demos, src_langs, languages=LANGS,
         scripts=("native", "roman"), tag=""):
    """Shared body of classes 1-3, and of class 4 (which is these at 3 shots)."""
    out = []
    for lang in languages:
        for script in scripts:
            tgt = lang if script == "native" else f"{lang}_rom"
            for src in (src_langs if task == "translation" else [tgt]):
                pool = usable(WORDS, task, src=src, lang=lang)
                if len(pool) <= n_demos:
                    continue
                for i, t in enumerate(pool):
                    demos = [w for j, w in enumerate(pool) if j != i][:n_demos]
                    args = ((t, demos, src, tgt) if task == "translation"
                            else (t, demos, tgt))
                    out.append(rec(
                        id=f"{task}{tag}_{src}_{lang}_{script}_{t['en']}",
                        task=f"{task}{tag}", src_lang=src, language=lang,
                        script=script, cognate=t.get("cognate", False),
                        n_demos=n_demos, target_label=f"{lang}_{script}",
                        prompt=TEMPLATES[task](*args),
                        answers=answers_for(t, lang, src)))
    return out


# ===========================================================================
# CLASS 0 -- Wendler's own example, verbatim
# ===========================================================================

def class0_wendler():
    """Correctness check, not an experiment. English should rise in middle
    layers and 花 win at the end. Gemma 3 is not Llama-2, so the curve will
    differ, but the shape should be recognisable. If it is not, stop and debug
    before running anything else."""
    return [rec(
        id="wendler_zh_reference", task="translation", src_lang="fr",
        language="zh", script="native", target_label="zh_native",
        prompt=('Français: "vertu" - 中文: "德"\n'
                'Français: "siège" - 中文: "座"\n'
                'Français: "neige" - 中文: "雪"\n'
                'Français: "montagne" - 中文: "山"\n'
                'Français: "fleur" - 中文: "'),
        answers={"en": "flower", "zh_native": "花", "en_control": "window"})]


# ===========================================================================
# CLASS 1 -- translation
# ===========================================================================

def class1_translation(n_demos=4, src_langs=("fr", "en"), languages=LANGS):
    """French holds the source constant so English is never the INPUT
    language: any English appearing mid-stack is internal, which is the whole
    point of Wendler's design. English is included as a second source because
    base models behave very differently there -- with a Latin language tag they
    tend to copy the English word rather than translate."""
    return core("translation", n_demos, src_langs, languages)


# ===========================================================================
# CLASS 2 -- cloze
# ===========================================================================

def class2_cloze(n_demos=4, languages=LANGS):
    """A sentence with one blank and one natural answer. Unlike translation,
    the model is never shown the answer in another language, so any English
    mid-stack cannot be explained by an English source word in the prompt."""
    return core("cloze", n_demos, None, languages)


# ===========================================================================
# CLASS 3 -- repetition
# ===========================================================================

def class3_repetition(n_demos=4, languages=LANGS):
    """Wendler's syntactic control. No translation is required, so an English
    hump here cannot be about meaning -- it would indicate something about the
    readout, or about how the model represents the token itself."""
    return core("repetition", n_demos, None, languages)


# ===========================================================================
# CLASS 4 -- three-shot versions of 1-3
# ===========================================================================

def class4_three_shot(n_demos=3, src_langs=("fr", "en"), languages=LANGS):
    """Shot count is a real variable, not a detail: a base model copies the
    English word zero-shot but translates once it has demonstrations. Comparing
    against the 4-shot sets tells you whether your phase structure is stable or
    an artifact of prompt length."""
    out = core("translation", n_demos, src_langs, languages, tag="_3shot")
    out += core("cloze", n_demos, None, languages, tag="_3shot")
    out += core("repetition", n_demos, None, languages, tag="_3shot")
    return out


# ===========================================================================
# CLASS 5 -- task conflict (demos one language, tag the next)
# ===========================================================================

def class5_lang_swap(n_demos=3, src_langs=("fr", "en")):
    """Cycle: ta -> hi, hi -> ml, ml -> ta.

    The in-context demonstrations say one language; the explicit tag says
    another. Watching WHERE the model resolves that conflict separates the
    ICL-induced task representation from the explicit instruction. Neither
    Wendler nor RomanLens does this, and it gives a sharper handle on the final
    phase: late resolution means language selection is a shallow operation
    sitting on top of an already-settled concept.

    `demo_language` records the misleading side so you can group by it.
    """
    out = []
    for src in src_langs:
        pool = usable(WORDS, "translation", src=src)
        for i, t in enumerate(pool):
            demos = [w for j, w in enumerate(pool) if j != i][:n_demos]
            for demo_lang in LANGS:
                final = SWAP[demo_lang]
                for script in ("native", "roman"):
                    d_tag = (demo_lang if script == "native"
                             else f"{demo_lang}_rom")
                    f_tag = final if script == "native" else f"{final}_rom"
                    lines = [f'{TAG[src]}: "{surface(d, src)}" - '
                             f'{TAG[d_tag]}: "{surface(d, d_tag)}"'
                             for d in demos]
                    lines.append(f'{TAG[src]}: "{surface(t, src)}" - '
                                 f'{TAG[f_tag]}: "')
                    a = answers_for(t, final, src)
                    a[f"{demo_lang}_native"] = t[demo_lang][0]
                    out.append(rec(
                        id=f"swap_{src}_{demo_lang}to{final}_{script}_{t['en']}",
                        task="lang_swap", src_lang=src, language=final,
                        demo_language=demo_lang, script=script,
                        target_label=f"{final}_{script}",
                        prompt="\n".join(lines), answers=a))
    return out


# ===========================================================================
# CLASS 6 -- verse continuation from the epics (memorisation)
# ===========================================================================

def class6_epic(path, language, task=None, prefix_words=8, max_items=40,
                min_words=None, seed=0, min_purity=0.85, clean=True,
                verbose=True):
    """Kamba Ramayanam in Tamil, Ramcharitmanas in Hindi: the same story at two
    resource levels, a control most of this literature cannot get.

    Give `prefix_words` words, ask for the next one. SWEEP the prefix length --
    that sweep IS the measurement. Memorised text shows a sharp accuracy jump
    once the prefix uniquely identifies the passage; genuine generalisation
    improves smoothly.

    There is no English counterpart for "the next word of this verse", so `en`
    is absent and only `en_control` is tracked. That is the right baseline: it
    asks whether English tokens rise mid-stack even when no English answer
    exists. The script-mass curves the pipeline records automatically are the
    other half.

    Prediction worth stating up front: verbatim recall involves no translation,
    so it should show NO English detour. A memorised Tamil verse that still
    showed one would be genuinely surprising.
    """
    task = task or f"epic_{language}"
    rng = random.Random(seed)
    # A line needs more than prefix_words tokens to yield an answer. A fixed
    # minimum would silently discard short verses -- wrong for poetry.
    min_words = min_words if min_words is not None else prefix_words + 1

    if clean:
        # Archive.org scans carry <200> markers, page numbers, running headers
        # and stray Latin from imperfect OCR. Without this the scored "next
        # word" can be noise, and you would be measuring the scanner.
        from utils.textclean import clean_lines
        raw, _ = clean_lines(path, language, min_purity=min_purity,
                             min_words=min_words, verbose=verbose)
    else:
        raw = [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

    lines = [l for l in raw
             if len(l.split()) > prefix_words and len(l.split()) >= min_words]
    if not lines:
        longest = max((len(l.split()) for l in raw), default=0)
        print(f"[warn] {path}: no line exceeds {prefix_words} words "
              f"(longest has {longest}). Lower --prefix-words, or check the "
              f"file is one verse per line and not one word per line.")
        return []
    rng.shuffle(lines)

    out = []
    for k, line in enumerate(lines[:max_items]):
        words = line.split()
        out.append({
            "id": f"{task}_{language}_{prefix_words}w_{k}",
            "task": task, "src_lang": language, "language": language,
            "script": "native", "prefix_words": prefix_words,
            "source_file": os.path.basename(path),
            "target_label": f"{language}_native",
            "prompt": " ".join(words[:prefix_words]) + " ",
            "answers": {f"{language}_native": words[prefix_words],
                        "en_control": rng.choice(EN_CONTROLS)},
            "collision_labels": [f"{language}_native"]})
    return out


# ===========================================================================
# CLASS 7 -- technical / clinical completion from FLORES-200
# ===========================================================================

DOMAIN_KEYWORDS = {
    "medical": ["patient", "disease", "cells", "cell ", "infection", "virus",
                "treatment", "symptom", "vaccine", "clinical", "protein",
                "blood", "cancer", "bacteria", "immune", "diagnos",
                "therapy", "surgery", "medicine", "gene"],
    "science": ["particle", "quantum", "molecule", "atom", "energy",
                "experiment", "theory", "measure", "physics", "chemical",
                "orbit", "electron", "temperature", "hypothesis", "equation"],
}


def class7_technical(sents, domain, prefix_words=(8,), max_items=40, seed=0):
    """sents: {lang: [sentences]}, index-aligned across languages.

    Why FLORES: it is N-WAY PARALLEL, so the same sentence exists in English,
    Hindi, Tamil and Malayalam. "Does the model complete this in Tamil as well
    as in English" is then a controlled comparison rather than two unrelated
    texts -- which a scraped corpus cannot give you.

    Why medical rather than pure maths: medicine has genuine Indic coverage,
    just thinner, so you can separate "low resource" from "absent entirely". A
    null on pure maths would be uninterpretable -- you could not tell whether
    the model lacks the concept or the language never discusses it.

    Filtering on the ENGLISH side and taking the aligned Indic sentence keeps
    the subset identical across languages.
    """
    kws = DOMAIN_KEYWORDS[domain]
    keep = [i for i, s in enumerate(sents["en"])
            if any(k in s.lower() for k in kws)]
    print(f"[{domain}] {len(keep)}/{len(sents['en'])} sentences matched")
    if not keep:
        return []

    rng = random.Random(seed)
    out = []
    for lg in sents:
        for k in prefix_words:
            taken = 0
            for i in keep:
                words = sents[lg][i].split()
                if len(words) <= k + 1:
                    continue
                out.append({
                    "id": f"{domain}_{lg}_{k}w_{i}",
                    "task": f"{domain}_completion", "src_lang": lg,
                    "language": lg, "script": "native",
                    "prefix_words": k, "flores_index": i,
                    "target_label": f"{lg}_native",
                    "prompt": " ".join(words[:k]) + " ",
                    "answers": {f"{lg}_native": words[k],
                                "en_control": rng.choice(EN_CONTROLS)},
                    "collision_labels": [f"{lg}_native"]})
                taken += 1
                if taken >= max_items:
                    break
    return out


# ===========================================================================
# CLASS 8 -- script-shaped gibberish (null)
# ===========================================================================

CONSONANTS = {
    "hi": "कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह",
    "ta": "கஙசஞடணதநபமயரலவழளறன",
    "ml": "കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ",
}
VOWELS = {"hi": "ािीुूेैोौ", "ta": "ாிீுூெேைொோௌ",
          "ml": "ാിീുൂെേൈൊോൌ"}


def class8_gibberish(languages=LANGS, n=30, words_per=10, seed=0):
    """Real characters of the target script assembled into non-words.

    A null with realistic surface form: if Tamil-script gibberish still shows
    an English hump mid-stack, that hump is not about concepts at all. Pair it
    with the random-vector nulls in analysis/controls.py, which attack the same
    question from the activation side rather than the input side.
    """
    rng = random.Random(seed)
    out = []
    for lang in languages:
        cons, vow = CONSONANTS[lang], VOWELS[lang]
        for k in range(n):
            words = ["".join(rng.choice(cons) + rng.choice(vow)
                             for _ in range(rng.randint(2, 3)))
                     for _ in range(words_per)]
            out.append({
                "id": f"gibberish_{lang}_{k}",
                "task": "gibberish", "src_lang": lang, "language": lang,
                "script": "native", "target_label": f"{lang}_native",
                "prompt": " ".join(words[:-1]) + " ",
                "answers": {f"{lang}_native": words[-1],
                            "en_control": rng.choice(EN_CONTROLS)},
                "collision_labels": [f"{lang}_native"]})
    return out


# ---------------------------------------------------------------------------
# Registry: classes that need no external data
# ---------------------------------------------------------------------------

BUILDERS = {
    "class0_wendler": class0_wendler,
    "class1_translation": class1_translation,
    "class2_cloze": class2_cloze,
    "class3_repetition": class3_repetition,
    "class4_three_shot": class4_three_shot,
    "class5_lang_swap": class5_lang_swap,
    "class8_gibberish": class8_gibberish,
}