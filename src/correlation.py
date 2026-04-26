"""
correlation.py
==============
Statistical analysis:
  1. Spearman rank correlation between PMI and Dependency Distance
     — computed per language
  2. Linear Mixed-Effects Model (LMM) across all languages
     — PMI and typology as fixed effects, language as random effect

Why Spearman (not Pearson)?
    Both PMI and DD are heavily right-skewed in natural language corpora.
    Spearman's rho is rank-based and makes no normality assumptions,
    making it more appropriate and robust to outliers.

Why a Mixed-Effects Model?
    Observations within the same language are not independent.
    The LMM treats language as a random effect, controlling for
    cross-language variation while estimating the overall PMI effect.

Usage:
    python src/correlation.py
"""

import pandas as pd
import numpy  as np
from scipy.stats          import spearmanr
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

from load_corpus   import load_sentences, LANGUAGE_FILES, TYPOLOGY_MAP
from extract_pairs import extract_pairs
from compute_pmi   import compute_pmi, build_dataframe


def run_full_pipeline(min_count: int = 5) -> tuple[dict, pd.DataFrame]:
    """
    Run the full analysis pipeline across all languages.

    Returns
    -------
    all_results : dict
        Per-language Spearman correlation results.
    combined : pd.DataFrame
        Combined DataFrame of all languages (for LMM).
    """
    all_results = {}
    all_dfs     = []

    for lang, filepaths in LANGUAGE_FILES.items():
        print(f"\n{'='*50}")
        print(f"  Language: {lang}  |  Typology: {TYPOLOGY_MAP[lang]}")
        print(f"{'='*50}")

        sentences  = load_sentences(filepaths)
        pairs      = extract_pairs(sentences)
        pmi_scores = compute_pmi(pairs, min_count=min_count)
        typology   = TYPOLOGY_MAP[lang]
        df         = build_dataframe(pairs, pmi_scores, lang, typology)

        if len(df) < 10:
            print(f"  [skip] Not enough data for {lang}")
            continue

        corr, pval = spearmanr(df["PMI"], df["DD"])
        print(f"  [result] Spearman r = {corr:.4f},  p = {pval:.2e}")

        all_results[lang] = {
            "language":   lang,
            "typology":   typology,
            "spearman_r": round(corr, 6),
            "p_value":    pval,
            "n_pairs":    len(df),
        }
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    return all_results, combined


def fit_mixed_effects_model(combined: pd.DataFrame) -> object:
    """
    Fit a Linear Mixed-Effects Model:
        DD ~ PMI + C(typology) + (1 | language)

    Parameters
    ----------
    combined : pd.DataFrame
        Combined DataFrame from all languages.

    Returns
    -------
    Fitted statsmodels MixedLMResults object.
    """
    # Subsample if very large (speeds up fitting)
    if len(combined) > 500_000:
        combined = combined.sample(500_000, random_state=42)
        print(f"  [lmm] Subsampled to 500,000 rows")

    print(f"  [lmm] Fitting model on {len(combined):,} observations "
          f"across {combined['language'].nunique()} languages...")

    model  = smf.mixedlm(
        "DD ~ PMI + C(typology)",
        combined,
        groups=combined["language"]
    )
    fitted = model.fit(method="lbfgs")
    print(fitted.summary())
    return fitted


def print_results_table(all_results: dict) -> pd.DataFrame:
    """Print and return a formatted results DataFrame."""
    results_df = pd.DataFrame(all_results.values())
    results_df = results_df.sort_values("spearman_r")
    results_df["significant"] = results_df["p_value"] < 0.05

    print("\n" + "="*70)
    print("SPEARMAN CORRELATION RESULTS — PMI vs DEPENDENCY DISTANCE")
    print("="*70)
    print(results_df[["language","typology","spearman_r",
                       "p_value","significant","n_pairs"]]
          .to_string(index=False))
    print("="*70)
    return results_df


if __name__ == "__main__":
    # Run full pipeline
    all_results, combined = run_full_pipeline(min_count=5)

    # Print results table
    results_df = print_results_table(all_results)

    # Save to CSV
    results_df.to_csv("results/correlation_results.csv", index=False)
    combined[["language","typology","PMI","DD"]]\
        .to_csv("results/combined_data.csv", index=False)
    print("\nSaved: results/correlation_results.csv")
    print("Saved: results/combined_data.csv")

    # Fit mixed-effects model
    print("\n--- Linear Mixed-Effects Model ---")
    fitted = fit_mixed_effects_model(combined)
