"""
data_loader.py
==============
Responsible for turning raw per-category CSV files into one combined,
correctly-labeled, correctly-sampled dataset.

THE CORE PROBLEM THIS FILE SOLVES
The raw flow-based CSVs are NOT labeled on a per-row basis -- every row's
`Label` column literally says "NeedManualLabel" straight out of the
CICFlowMeter tool. The dataset's README is explicit about this: labels
must be assigned by the researcher based on which folder/file a row came
from. So "loading the data" and "labeling the data" are the same step
here, not two separate ones -- we relabel every row the moment we load
its source file, based on which category that file represents.

THE SECOND PROBLEM THIS FILE SOLVES
Real-world attack data is scarce relative to benign data (that's the
whole point of anomaly detection -- attacks are rare events). Our raw
files reflect this: some categories have only ~2,000-4,000 rows, while
benign traffic could in principle be almost unlimited. The `load_category`
and `build_dataset` functions here implement a SAMPLER: rather than using
every row in every file (which would blow past what's practical, and
wouldn't match the realistic ~97/3 imbalance the professor's spec wants),
we draw a controlled, seeded random sample from each category's pool.
"""

import os
import numpy as np
import pandas as pd

import config


def load_category(category, n_target, seed_offset):
    """
    Load and pool every file mapped to `category` in config.FILE_MAP,
    relabel all rows to `category`, then randomly sample n_target rows.

    Parameters
    ----------
    category : str
        One of "Benign" or an entry in config.ATTACK_CATEGORIES.
    n_target : int
        How many rows to sample from this category's pooled data.
    seed_offset : int
        Added to the global seed so each category gets a DIFFERENT
        (but still reproducible) random sample. Without an offset,
        every category would draw "the same" random indices relative
        to its own pool, which is harmless here but is good practice
        to avoid subtle correlated-sampling artifacts.

    Returns
    -------
    pandas.DataFrame
        n_target rows (or fewer, if the pool is smaller than n_target),
        all labeled with `category` in the Label column.

    WHY POOL MULTIPLE FILES BEFORE SAMPLING?
    Some categories (Benign, DoS-HTTP_Flood) are split across multiple
    files that represent the SAME underlying traffic type. Pooling them
    first means our random sample can draw from the full combined pool
    rather than being artificially restricted to whichever single file
    we happened to list first -- this gives a more representative sample
    and lets us reach larger sample sizes for categories with more total
    real data available.
    """
    fnames = config.FILE_MAP.get(category, [])
    frames = []
    for fname in fnames:
        fpath = os.path.join(config.DATA_DIR, fname)
        if os.path.exists(fpath):
            frames.append(pd.read_csv(fpath))
        else:
            print(f"  [WARNING] {category}: expected file not found -> {fpath}")

    if not frames:
        raise FileNotFoundError(
            f"No files found for category '{category}'. Check config.FILE_MAP "
            f"and config.DATA_DIR."
        )

    pooled = pd.concat(frames, ignore_index=True)
    pooled_n = len(pooled)

    # THE LABELING STEP: overwrite whatever placeholder value was in the
    # Label column (typically "NeedManualLabel") with the real category,
    # since we know it deterministically from which file(s) this data
    # came from.
    pooled[config.LABEL_COL] = "Benign" if category == "Benign" else category

    if pooled_n > n_target:
        sample = pooled.sample(n=n_target, random_state=config.SEED + seed_offset)
    else:
        # If we're asking for more rows than exist, just take everything
        # and warn -- silently returning fewer rows than requested could
        # otherwise go unnoticed and skew your reported class balance.
        sample = pooled
        if pooled_n < n_target:
            print(f"  [NOTE] {category}: pool only has {pooled_n} rows, "
                  f"fewer than requested {n_target} -- using all of them.")

    print(f"  [LOADED] {category}: pooled {pooled_n} rows from "
          f"{len(frames)} file(s) -> sampled {len(sample)}")
    return sample


def build_dataset(seed=None):
    """
    Build the full labeled, sampled dataset: one Benign pool plus five
    attack categories, combined and shuffled.

    Parameters
    ----------
    seed : int or None
        Overrides config.SEED for this call. Used by the multi-seed
        robustness check (see evaluate.py) to get a genuinely different
        random composition on each run without editing config.py.

    Returns
    -------
    pandas.DataFrame
        The full combined dataset, shuffled, with a correct Label column.

    HOW THE ATTACK BUDGET IS SPLIT ACROSS CATEGORIES
    config.ATTACK_TARGET_TOTAL (e.g. 1,200) is divided as evenly as
    possible across the 5 attack categories. If it doesn't divide evenly
    (1,200 / 5 = 240 exactly, but this logic generalizes to totals that
    don't divide cleanly), the remainder is distributed one extra row
    at a time to the first few categories, so no category is shorted
    by more than one row relative to the others. This satisfies the
    spec's "spread roughly uniformly across the five categories" while
    still handling arbitrary totals correctly.
    """
    active_seed = seed if seed is not None else config.SEED
    print(f"Building dataset (seed = {active_seed})")

    frames = [load_category("Benign", config.BENIGN_TARGET, seed_offset=0)]

    n_categories = len(config.ATTACK_CATEGORIES)
    base = config.ATTACK_TARGET_TOTAL // n_categories
    remainder = config.ATTACK_TARGET_TOTAL - base * n_categories

    for i, cat in enumerate(config.ATTACK_CATEGORIES):
        n = base + (1 if i < remainder else 0)
        frames.append(load_category(cat, n, seed_offset=i + 1))

    df = pd.concat(frames, ignore_index=True)

    # Shuffle so categories aren't grouped in contiguous blocks -- this
    # matters because some downstream operations (e.g. a quick .head()
    # sanity check, or if you ever DON'T stratify a split) could otherwise
    # accidentally see a skewed slice of the data.
    df = df.sample(frac=1, random_state=active_seed).reset_index(drop=True)

    print(f"\nFinal dataset shape: {df.shape}")
    counts = df[config.LABEL_COL].value_counts()
    print(counts)
    total = len(df)
    benign_pct = counts.get("Benign", 0) / total * 100
    attack_pct = 100 - benign_pct
    print(f"\nBenign: {benign_pct:.1f}%  |  Attack: {attack_pct:.1f}%  "
          f"(spec target: ~97-98% / ~2-3%)")

    return df
