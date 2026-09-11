"""All prompt-class builders in one place.

Word lists live in promptutils/words.py. Every class here reads from it.

    class1_translation    translation, native + romanised
    class2_cloze          cloze completion
    class3_repetition     copy task (syntactic control)
    class4_lang_swap      demos in one language, tag in the next
    class5_gibberish      script-shaped nonsense (null)
    class6_roman_to_roman romanised Indic -> romanised Indic

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
                                          just as high?
  src             the source-side word    Wendler reports input-language
                                          probability stays ~0; verify it

Only `collision_labels` must be mutually distinct. Filtering on all six would
destroy the sample, and controls stay readable when they partly overlap.

Romanised = the native script transliterated into Latin letters, not the
English word.
"""

import os
import random

from .words import EN_CONTROLS_RANDOM, WORDS


def _load_controls():
    """promptutils/controls.json from build_controls.py, if it exists.

    Maps each English word to one matched on unigram frequency and on the size
    of its Start(w) set. Without the file we fall back to EN_CONTROLS_RANDOM,
    a fixed list matched on nothing: enough to show whether an arbitrary
    English word peaks as high as the correct one, but a near-1 ratio would be
    uninterpretable because these words might simply be frequent.
    """
    import json
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "controls.json")
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
SWAP = {"ta": "hi", "hi": "ml", "ml": "ta"}      # class 4 cycle
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
         "en_control": CONTROLS.get(w["en"], rng.choice(EN_CONTROLS_RANDOM))}
    a[f"{PIVOT[language]}_native"] = w[PIVOT[language]][0]
    if src_lang and src_lang != "en":
        a["src"] = surface(w, src_lang)
    return a


def rec(**kw):
    kw.setdefault("collision_labels", ["en", kw["target_label"]])
    return kw


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

    That is right for a demonstration, but the target line must stop at the
    opening quote so the model supplies the word. The answer sits between the
    last two quotes, so rsplit finds it without knowing the answer label in
    each language ("Answer", "பதில்", "答案").
    """
    parts = frame.rsplit('"', 2)
    return parts[0] + '"' if len(parts) == 3 else frame.rstrip() + ' "'


def t_cloze(target, demos, lang):
    """Frames live in words.py. In the romanised condition the frame stays in
    native script but the ANSWER is romanised.

    Demonstrations keep the frame's own answer; only the final line is opened.
    """
    base = lang[:-4] if lang.endswith("_rom") else lang
    if lang.endswith("_rom"):
        # the frame's built-in answer is native script; swap in the romanised
        # form so the demonstrated output script matches what we ask for
        lines = [cloze_open(d["cloze"][base]) + surface(d, lang) + '".'
                 for d in demos]
    else:
        lines = [d["cloze"][base] for d in demos]
    lines.append(cloze_open(target["cloze"][base]))
    return "\n".join(lines)


TEMPLATES = {"translation": t_translation, "repetition": t_repetition,
             "cloze": t_cloze}


def usable(words, task, src=None, lang=None):
    """Words that have every surface form this prompt will reference.

    Source coverage is incomplete in the RomanLens tables (French has 117 of
    132, German 120), so an unfiltered translation prompt yields lines like
    'Français: "" - தமிழ்: "'. A romanised source lives in the second element
    of the (native, roman) tuple rather than a top-level column, so it needs a
    different check. Cloze needs a frame in the target language.
    """
    out = []
    for w in words:
        if task == "translation" and src:
            if src.endswith("_rom"):
                base = src[:-4]
                if not (w.get(base) and str(w[base][1]).strip()):
                    continue
            elif not (w.get(src) or "").strip():
                continue
        if task == "cloze" and lang and not (w.get("cloze", {}) or {}).get(lang):
            continue
        out.append(w)
    return out


def core(task, n_demos, src_langs, languages=LANGS,
         scripts=("native", "roman"), tag="", id_prefix=None):
    """Shared body of classes 1-3 and 6."""
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
                    stem = id_prefix or f"{task}{tag}"
                    out.append(rec(
                        id=f"{stem}_{src}_{lang}_{script}_{t['en']}",
                        task=f"{task}{tag}", src_lang=src, language=lang,
                        script=script, cognate=t.get("cognate", False),
                        n_demos=n_demos, target_label=f"{lang}_{script}",
                        prompt=TEMPLATES[task](*args),
                        answers=answers_for(t, lang, src)))
    return out


# ===========================================================================
# CLASS 1 -- translation
# ===========================================================================

def class1_translation(n_demos=4, src_langs=("fr", "en"), languages=LANGS):
    """French holds the source constant so English is never the INPUT
    language: any English appearing mid-stack is internal, which is the point
    of Wendler's design. English is included as a second source because base
    models behave very differently there -- with a Latin language tag they tend
    to copy the English word rather than translate."""
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
    hump here cannot be about meaning."""
    return core("repetition", n_demos, None, languages)


# ===========================================================================
# CLASS 4 -- task conflict (demos one language, tag the next)
# ===========================================================================

def class4_lang_swap(n_demos=4, src_langs=("fr", "en")):
    """Cycle: ta -> hi, hi -> ml, ml -> ta.

    The demonstrations say one language; the explicit tag says another. The
    layer at which the model resolves that conflict separates the ICL-induced
    task representation from the explicit instruction. Neither Wendler nor
    RomanLens does this.

    `demo_language` records the misleading side so results can be grouped by it.
    """
    out = []
    for src in src_langs:
        pool = usable(WORDS, "translation", src=src)
        if len(pool) <= n_demos:
            continue
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
                        n_demos=n_demos, target_label=f"{final}_{script}",
                        prompt="\n".join(lines), answers=a))
    return out


# ===========================================================================
# CLASS 5 -- script-shaped gibberish (null)
# ===========================================================================

CONSONANTS = {
    "hi": "कखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह",
    "ta": "கஙசஞடணதநபமயரலவழளறன",
    "ml": "കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ",
}
VOWELS = {"hi": "ािीुूेैोौ", "ta": "ாிீுூெேைொோௌ",
          "ml": "ാിീുൂെേൈൊോൌ"}


def class5_gibberish(languages=LANGS, n=30, words_per=10, seed=0):
    """Real characters of the target script assembled into non-words.

    An input-side null with realistic surface form: if Tamil-script gibberish
    still shows an English hump mid-stack, that hump is not about concepts.
    There is no meaningful correct answer here, so read the SCRIPT-MASS curves
    rather than P(lang).
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
                "script": "native", "n_demos": 0,
                "target_label": f"{lang}_native",
                "prompt": " ".join(words[:-1]) + " ",
                "answers": {f"{lang}_native": words[-1],
                            "en_control": rng.choice(EN_CONTROLS_RANDOM)},
                "collision_labels": [f"{lang}_native"]})
    return out


# ===========================================================================
# CLASS 6 -- romanised Indic -> romanised Indic
# ===========================================================================

def class6_roman_to_roman(n_demos=4):
    """Romanised Hindi -> romanised Tamil, and the other pairings.

    Nothing in the prompt is in native script, so native-script probability
    mid-stack has to be internal. This is the mirror of Wendler's logic: he
    uses a non-English source so English cannot be copied; here the whole
    prompt is Latin so native script cannot be copied.

    Note the English curve changes meaning in this class. Latin-script mass is
    high throughout by construction, so the informative quantity is P(en)
    against en_control rather than against the script-mass baseline.
    """
    out = []
    for src, tgt in (("hi", "ta"), ("ta", "ml"), ("ml", "hi")):
        out += core("translation", n_demos, (f"{src}_rom",), (tgt,),
                    scripts=("roman",), tag="_r2r",
                    id_prefix=f"r2r_{src}to{tgt}")
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILDERS = {
    "class1_translation": class1_translation,
    "class2_cloze": class2_cloze,
    "class3_repetition": class3_repetition,
    "class4_lang_swap": class4_lang_swap,
    "class5_gibberish": class5_gibberish,
    "class6_roman_to_roman": class6_roman_to_roman,
}

# Classes whose demonstration count run_all sweeps. class4_lang_swap accepts
# n_demos but is held at 4; class5 has no demonstrations at all.
SHOT_SWEPT = ("class1_translation", "class2_cloze", "class3_repetition",
              "class6_roman_to_roman")