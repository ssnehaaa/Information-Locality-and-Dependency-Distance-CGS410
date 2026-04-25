"""
load_corpus.py
==============
Load and parse SUD (Surface-Syntactic Universal Dependencies)
.conllu files using the `conllu` Python library.

Usage:
    from src.load_corpus import load_sentences, LANGUAGE_FILES, TYPOLOGY_MAP
"""

import conllu

# ── Language → file mapping ────────────────────────────────────────────────
# Place all .conllu files in the same directory before running.
LANGUAGE_FILES = {
    "English":  ["en_gum-sud-train.conllu"],
    "French":   [
        "fr_gsd-sud-train_A.conllu",
        "fr_gsd-sud-train_B.conllu",
        "fr_gsd-sud-train_C.conllu",
        "fr_gsd-sud-train_D.conllu",
        "fr_gsd-sud-train_E.conllu",
    ],
    "German":   ["de_gsd-sud-train.conllu"],
    "Hindi":    ["hi_hdtb-sud-train.conllu"],
    "Japanese": ["ja_gsd-sud-train.conllu"],
    "Persian":  ["fa_seraji-sud-train.conllu"],
    "Russian":  ["ru_syntagrus-sud-train.conllu"],
    "Turkish":  ["tr_kenet-sud-train.conllu"],
    "Chinese":  ["zh_gsd-sud-train.conllu"],
}

# ── Typological classification ─────────────────────────────────────────────
TYPOLOGY_MAP = {
    "English":  "SVO",
    "French":   "SVO",
    "Chinese":  "SVO",
    "German":   "Mixed",
    "Russian":  "Mixed",
    "Hindi":    "SOV",
    "Japanese": "SOV",
    "Persian":  "SOV",
    "Turkish":  "SOV",
}


def load_sentences(filepaths: list[str]) -> list:
    """
    Parse one or more .conllu files and return a flat list of sentences.

    Parameters
    ----------
    filepaths : list[str]
        Paths to one or more .conllu files belonging to the same language.
        (French is split into A/B/C/D/E and must be passed together.)

    Returns
    -------
    list of conllu.TokenList
        All parsed sentences across all files.
    """
    all_sentences = []
    for path in filepaths:
        with open(path, "r", encoding="utf-8") as f:
            parsed = conllu.parse(f.read())
            all_sentences.extend(parsed)
        print(f"  [load] {path}: {len(parsed):,} sentences")

    print(f"  [load] Total: {len(all_sentences):,} sentences\n")
    return all_sentences


if __name__ == "__main__":
    # Quick test — loads English only
    sents = load_sentences(LANGUAGE_FILES["English"])
    print(f"First sentence: {[tok['form'] for tok in sents[0]]}")
