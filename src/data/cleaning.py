import random
import pandas as pd


def split_indices(
    n: int,
    test_fraction: float,
    seed: int,
    strata: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    """Deterministic split of `range(n)` into (train_idx, test_idx).

    When `strata` is provided (length `n`), the split is stratified so the
    fraction held out is approximately `test_fraction` within every stratum
    (e.g. equal treatment-arm balance across train/test). Otherwise the
    split is a uniform random shuffle.

    Args:
        n: Total number of items to split.
        test_fraction: Fraction held out per stratum (or overall when
            `strata` is None). Rounded to the nearest integer per bucket.
        seed: Seed for the underlying `random.Random` for determinism.
        strata: Optional per-item stratum labels driving stratified
            sampling. Must have length `n` if provided.

    Returns:
        A `(train_idx, test_idx)` pair of disjoint index lists whose union
        equals `range(n)`. Both lists are shuffled.

    Raises:
        ValueError: If `strata` is provided but its length does not match `n`.
    """
    rng = random.Random(seed)

    if strata is None:
        idx = list(range(n))
        rng.shuffle(idx)
        n_test = int(round(n * test_fraction))
        return idx[n_test:], idx[:n_test]

    if len(strata) != n:
        raise ValueError(f"strata length {len(strata)} does not match n={n}.")

    by_stratum: dict[str, list[int]] = {}
    for i, stratum in enumerate(strata):
        by_stratum.setdefault(stratum, []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for stratum in sorted(by_stratum):
        bucket = by_stratum[stratum]
        rng.shuffle(bucket)
        n_test = int(round(len(bucket) * test_fraction))
        test_idx.extend(bucket[:n_test])
        train_idx.extend(bucket[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def load_data(filepath: str) -> tuple[pd.DataFrame, dict]:
    """Load an RCT data file with a two-row header.

    File layout:
        Row 0: short variable codes (used as DataFrame column names).
        Row 1: long-form survey questions/labels.
        Row 2+: subject responses.

    Args:
        filepath: Path to a `.csv` or `.xlsx` file following the convention above.

    Returns:
        Tuple `(data, var_labels)` where `data` is the subject-response
        DataFrame keyed by short variable codes, and `var_labels` maps each
        short code to its long-form label.

    Raises:
        ValueError: If `filepath` does not end in `.csv` or `.xlsx`.
    """
    if filepath.endswith(".csv"):
        raw = pd.read_csv(filepath, header=0)
    elif filepath.endswith(".xlsx"):
        raw = pd.read_excel(filepath, header=0)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")

    var_labels = raw.iloc[0].to_dict()
    data = raw.iloc[1:].reset_index(drop=True)
    return data, var_labels
