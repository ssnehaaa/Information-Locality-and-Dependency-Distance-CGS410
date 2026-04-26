"""
compute_pmi.py
==============
Compute Pointwise Mutual Information (PMI) for head-dependent
word pairs extracted from dependency trees.

Formula:
    PMI(h, d) = log2[ P(h,d) / (P(h) * P(d)) ]

Where:
    P(h,d) = count(h,d) / total_arcs      [joint probability]
    P(h)   = count(h as head) / total      [marginal: head]
    P(d)   = count(d as dep)  / total      [marginal: dependent]

Interpretation:
    PMI > 0  : pair co-occurs MORE than chance  → strong association
    PMI = 0  : pair co-occurs exactly at chance
    PMI < 0  : pair co-occurs LESS than chance  → weak association

Design choices:
    - min_count=5 filters rare pairs to suppress noisy PMI estimates
      (pairs seen fewer than 5 times have unreliable probability estimates)
    - PMI is computed at the TYPE level (over word forms, not tokens)

Usage:
    from src.compute_pmi import compute_pmi, build_dataframe
"""

import math
import pandas as pd
from collections import Counter


def compute_pmi(
    pairs: list[tuple],
    min_count: int = 5
) -> dict[tuple, float]:
    """
    Compute PMI for every (head, dependent) word pair.

    Parameters
    ----------
    pairs : list of (str, str, int)
        Output of extract_pairs.extract_pairs()
    min_count : int
        Minimum number of times a pair must appear to be included.
        Default = 5 (filters hapax legomena and near-hapaxes).

    Returns
    -------
    dict mapping (head_word, dep_word) -> PMI score (float)
    """
    # Count co-occurrences
    pair_counts = Counter((h, d) for h, d, _ in pairs)
    head_counts = Counter(h       for h, d, _ in pairs)
    dep_counts  = Counter(d       for h, d, _ in pairs)
    total       = sum(pair_counts.values())

    pmi_scores = {}
    skipped    = 0

    for (h, d), c in pair_counts.items():
        # Filter rare pairs
        if c < min_count:
            skipped += 1
            continue

        p_hd = c              / total
        p_h  = head_counts[h] / total
        p_d  = dep_counts[d]  / total

        # Avoid log(0) — should not happen given min_count > 0
        if p_h * p_d == 0:
            continue

        pmi_scores[(h, d)] = math.log2(p_hd / (p_h * p_d))

    print(f"  [pmi] {len(pmi_scores):,} pairs kept  |  "
          f"{skipped:,} pairs filtered (count < {min_count})")
    return pmi_scores


def build_dataframe(
    pairs:      list[tuple],
    pmi_scores: dict,
    language:   str,
    typology:   str
) -> pd.DataFrame:
    """
    Merge the raw pairs list with PMI scores into a tidy DataFrame.

    Parameters
    ----------
    pairs      : list of (head, dep, DD) tuples
    pmi_scores : dict from compute_pmi()
    language   : language name string (e.g. 'English')
    typology   : typology string ('SVO', 'SOV', or 'Mixed')

    Returns
    -------
    pd.DataFrame with columns: head, dep, DD, PMI, language, typology
    Only rows where PMI was computed (i.e. pair passed min_count) are kept.
    """
    rows = []
    for h, d, dd in pairs:
        pmi = pmi_scores.get((h, d))
        if pmi is not None:
            rows.append({
                "head":     h,
                "dep":      d,
                "DD":       dd,
                "PMI":      pmi,
                "language": language,
                "typology": typology,
            })

    df = pd.DataFrame(rows)
    print(f"  [df]  {len(df):,} rows in final DataFrame")
    return df


if __name__ == "__main__":
    from load_corpus   import load_sentences, LANGUAGE_FILES, TYPOLOGY_MAP
    from extract_pairs import extract_pairs

    lang   = "English"
    sents  = load_sentences(LANGUAGE_FILES[lang])
    pairs  = extract_pairs(sents)
    pmi    = compute_pmi(pairs)
    df     = build_dataframe(pairs, pmi, lang, TYPOLOGY_MAP[lang])

    print(f"\nTop 10 highest PMI pairs in {lang}:")
    print(df.nlargest(10, "PMI")[["head","dep","PMI","DD"]].to_string(index=False))

    print(f"\nTop 10 lowest PMI pairs in {lang}:")
    print(df.nsmallest(10, "PMI")[["head","dep","PMI","DD"]].to_string(index=False))
