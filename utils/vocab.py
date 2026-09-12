"""Vocabulary utilities: Start(w) token sets, collisions, Unicode scripts.

Wendler Appendix A.2: P(lang = l) is the summed probability of every token that
could be the FIRST token of the correct word in language l, considering the
word with and without a leading space.
"""

import unicodedata
from collections import defaultdict

SCRIPTS = ["LATIN", "DEVANAGARI", "TAMIL", "MALAYALAM", "CJK", "CYRILLIC",
           "ARABIC", "OTHER", "NONALPHA"]

# language code -> the Unicode block its native script lives in
LANG_SCRIPT = {"hi": "DEVANAGARI", "ta": "TAMIL", "ml": "MALAYALAM",
               "zh": "CJK", "en": "LATIN", "fr": "LATIN", "de": "LATIN"}


def token_script(s):
    """Dominant Unicode script of a token string, by first alphabetic char."""
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        for scr in ("LATIN", "DEVANAGARI", "TAMIL", "MALAYALAM",
                    "CYRILLIC", "ARABIC"):
            if name.startswith(scr):
                return scr
        if ("CJK" in name or name.startswith("HIRAGANA")
                or name.startswith("KATAKANA")):
            return "CJK"
        return "OTHER"
    return "NONALPHA"


class VocabIndex:
    """Precomputed index over the tokenizer vocabulary. Build once per model.

    Two products:
      by_first   first character -> token ids, so Start(w) is a small scan
                 rather than a pass over 262k entries per word
      by_script  Unicode block -> token ids, which is what makes the
                 script-mass observable a single gather-and-sum per layer
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        # SentencePiece marks a leading space with U+2581
        self.strings = [(p or "").replace("\u2581", " ") for p in pieces]

        self.by_first = defaultdict(list)
        self.by_script = defaultdict(list)
        for tid, s in enumerate(self.strings):
            if s:
                self.by_first[s[0]].append(tid)
            self.by_script[token_script(s)].append(tid)

    def start_token_ids(self, word):
        """Tokens that could begin `word`, with or without a leading space.

        Whitespace-only tokens are EXCLUDED. A bare space token is a prefix of
        every space-prefixed word, so including it puts one shared id in every
        word's start set and makes every pair collide -- which silently drops
        100% of items at the collision filter. It also carries no language
        information.
        """
        out = set()
        for variant in (word, " " + word):
            if not variant:
                continue
            for tid in self.by_first.get(variant[0], ()):
                s = self.strings[tid]
                if s.strip() and variant.startswith(s):
                    out.add(tid)
        return out

    def fertility(self, word):
        """Tokens per word, with a leading space."""
        return len(self.tokenizer.encode(" " + word, add_special_tokens=False))

    def is_single_token(self, word):
        return self.fertility(word) == 1 if word else False


def build_token_sets(index, answers):
    """{label: sorted token ids} for every surface form of an item."""
    out = {}
    for label, word in answers.items():
        ids = index.start_token_ids(word)
        if ids:
            out[label] = sorted(ids)
    return out


def has_collision(token_sets, labels=None):
    """True if two labels share a starting token, so they cannot be told apart
    from the next-token distribution.

    `labels` restricts the check to the labels carrying the measurement.
    Controls (en_control, a pivot language, the source word) are baselines, and
    filtering on all of them would destroy the sample.
    """
    labels = [l for l in (labels or list(token_sets)) if l in token_sets]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if set(token_sets[labels[i]]) & set(token_sets[labels[j]]):
                return True
    return False


def collision_detail(token_sets, index, labels=None):
    """Which tokens two labels share, for diagnosis."""
    labels = [l for l in (labels or list(token_sets)) if l in token_sets]
    out = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            shared = set(token_sets[labels[i]]) & set(token_sets[labels[j]])
            if shared:
                out.append((labels[i], labels[j],
                            [index.strings[t] for t in sorted(shared)][:6]))
    return out


def role_of(label, target_label=None):
    """Canonical role, so figures colour by meaning rather than by language.

    target_label MATTERS. Each item tracks the target's native form AND a third
    "pivot" language's native form, and both end in "_native". Without knowing
    which is the target, both collapse into the role "native" and the plotted
    curve becomes their MEAN -- halving the target curve against a pivot that
    sits near zero. Pass target_label so the pivot gets its own role.
    """
    if target_label and label != target_label and label.endswith("_native") \
            and target_label.endswith(("_native", "_roman")):
        return "pivot"
    if label == "en":
        return "en"
    if label == "en_control":
        return "en_ctrl"
    if label == "src":
        return "src"
    if label.startswith("script_"):
        return None                    # handled by the script-mass figure
    if label.endswith("_native"):
        return "native"
    if label.endswith("_roman"):
        return "roman"
    return "other"