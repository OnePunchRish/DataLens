"""
analytics/insights.py — Column statistics and AI insight engines.

Provides two public functions:
  - generate_insights    → per-column stats and auto-generated observations
  - generate_ai_insights → rule-based analyst-grade insight cards

No external APIs are used; all logic is derived directly from the data.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ── Shared types ───────────────────────────────────────────────────────────────

InsightCard = dict[str, Any]   # {severity, icon, headline, body, metric}
TrendInfo   = dict[str, Any]   # {direction, change_pct}

# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum rows needed to compute a meaningful trend
_TREND_MIN_ROWS = 10

# Threshold (%) to call a change an upward/downward trend
_TREND_UP_PCT   =  5.0
_TREND_DOWN_PCT = -5.0

# Thresholds for AI trend insight labels
_SURGE_PCT = 20.0
_DROP_PCT  = 20.0

# Standard-deviation multiplier for outlier detection
_OUTLIER_SIGMA = 3.0

# Minimum rows before flagging outliers
_OUTLIER_MIN_ROWS = 20

# Correlation coefficient thresholds
_CORR_VERY_STRONG = 0.85
_CORR_STRONG      = 0.70

# Skewness magnitude that triggers a distribution warning
_SKEW_HEAVY  = 2.0
_SKEW_NOTICE = 1.5

# Mean/median relative divergence threshold
_MEAN_MEDIAN_DIVERGENCE = 0.30

# How many top categories / numeric columns to profile per section
_MAX_CAT_PROFILE  = 4
_MAX_NUM_PROFILE  = 6
_MAX_AI_INSIGHTS  = 9
_AUTO_NOTES_LIMIT = 6

# Columns used when building correlation pairs
_CORR_MAX_COLS = 5


# ── Shared helpers ─────────────────────────────────────────────────────────────

def format_number(n: float | int) -> str:
    """
    Format a number for human-readable display.

    Applies magnitude suffixes (K, M) for large values and limits decimal
    places for floats.

    Examples:
        >>> format_number(1_500_000)
        '1.5M'
        >>> format_number(12345.678)
        '12.3K'
        >>> format_number(42)
        '42'
    """
    if isinstance(n, float):
        abs_n = abs(n)
        if abs_n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if abs_n >= 1_000:
            return f"{n / 1_000:.1f}K"
        return f"{n:,.2f}"
    return f"{n:,}"


def _make_insight(
    severity: str,
    icon: str,
    headline: str,
    body: str,
    metric: str | None = None,
) -> InsightCard:
    """
    Build a structured insight card dict.

    Args:
        severity: One of "positive", "negative", "warning", "info".
        icon:     Emoji or symbol shown beside the headline.
        headline: Short title (shown in bold).
        body:     One to two sentence explanation.
        metric:   Optional key figure displayed as a badge (e.g. "+12%").
    """
    return {
        "severity": severity,
        "icon":     icon,
        "headline": headline,
        "body":     body,
        "metric":   metric,
    }


# ── Trend detection ────────────────────────────────────────────────────────────

def _detect_trend(series: pd.Series) -> TrendInfo:
    """
    Compare the mean of the first half of `series` to the mean of the
    second half to infer a directional trend.

    Args:
        series: A numeric pandas Series (NaNs are dropped before analysis).

    Returns:
        Dict with keys "direction" ("up" | "down" | "flat") and
        "change_pct" (float, rounded to one decimal place).
    """
    clean = series.dropna()
    if len(clean) < _TREND_MIN_ROWS:
        return {"direction": "flat", "change_pct": 0.0}

    mid         = len(clean) // 2
    first_mean  = clean.iloc[:mid].mean()
    second_mean = clean.iloc[mid:].mean()

    if first_mean == 0:
        return {"direction": "flat", "change_pct": 0.0}

    change_pct = round((second_mean - first_mean) / abs(first_mean) * 100, 1)

    if change_pct > _TREND_UP_PCT:
        direction = "up"
    elif change_pct < _TREND_DOWN_PCT:
        direction = "down"
    else:
        direction = "flat"

    return {"direction": direction, "change_pct": change_pct}


# ── Column-level statistics ────────────────────────────────────────────────────

def _profile_numeric_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    """
    Compute descriptive statistics for a single numeric column.

    Args:
        df:  Source DataFrame.
        col: Column name (must be numeric).

    Returns:
        Dict containing mean, median, std, min, max, range, skew,
        non_null count, and trend info.  Returns an empty dict if the
        column contains no non-null values.
    """
    series = df[col].dropna()
    if series.empty:
        return {}

    return {
        "mean":     round(float(series.mean()),   3),
        "median":   round(float(series.median()), 3),
        "std":      round(float(series.std()),     3),
        "min":      round(float(series.min()),     3),
        "max":      round(float(series.max()),     3),
        "range":    round(float(series.max() - series.min()), 3),
        "skew":     round(float(series.skew()),    2),
        "non_null": int(series.count()),
        "trend":    _detect_trend(series),
    }


def _top_values_for_column(
    df: pd.DataFrame,
    col: str,
    n: int = 5,
) -> list[dict[str, Any]]:
    """
    Return the top `n` most frequent values for a categorical column.

    Args:
        df:  Source DataFrame.
        col: Column name.
        n:   Number of top values to return.

    Returns:
        List of dicts with keys "value", "count", and "pct" (percentage
        of total rows, rounded to one decimal place).
    """
    total  = len(df)
    counts = df[col].value_counts(dropna=True).head(n)
    return [
        {"value": str(v), "count": int(c), "pct": round(c / total * 100, 1)}
        for v, c in counts.items()
    ]


def _build_auto_notes(
    df: pd.DataFrame,
    numeric_insights: dict[str, dict],
    top_values: dict[str, list],
    categorical_cols: list[str],
) -> list[str]:
    """
    Generate plain-English observations from the profiled column data.

    Covers: most varied category, dominant value, trends, heavy skew, and
    high variability.  Returns at most `_AUTO_NOTES_LIMIT` notes.

    Args:
        df:                Source DataFrame.
        numeric_insights:  Output of `_profile_numeric_column` per column.
        top_values:        Output of `_top_values_for_column` per column.
        categorical_cols:  Names of categorical columns.

    Returns:
        List of human-readable observation strings.
    """
    notes: list[str] = []
    rows = len(df)

    # Most varied categorical column
    if categorical_cols:
        unique_counts = {c: df[c].nunique() for c in categorical_cols}
        most_varied   = max(unique_counts, key=unique_counts.get)
        notes.append(
            f'"{most_varied}" has the most variety — '
            f"{unique_counts[most_varied]:,} unique values across {rows:,} rows."
        )

    # Dominant value in the first categorical column
    if top_values:
        first_cat = next(iter(top_values))
        top = top_values[first_cat]
        if top:
            notes.append(
                f'The most common "{first_cat}" is '
                f'"{top[0]["value"]}" ({top[0]["pct"]}% of rows).'
            )

    # Per-column notes: trend, heavy skew, high variability (single pass)
    for col, info in numeric_insights.items():
        if not info:
            continue

        trend = info.get("trend", {})
        skew  = info.get("skew", 0) or 0
        mean  = info.get("mean", 0) or 0
        std   = info.get("std",  0) or 0

        if trend.get("direction") == "up":
            notes.append(
                f'"{col}" shows an upward trend '
                f'(+{trend["change_pct"]}% from first half to second half).'
            )
        elif trend.get("direction") == "down":
            notes.append(
                f'"{col}" shows a downward trend '
                f'({trend["change_pct"]}% from first half to second half).'
            )

        if abs(skew) > _SKEW_NOTICE:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            notes.append(
                f'"{col}" is heavily skewed {direction} (skewness = {skew}), '
                f"suggesting outliers or a long tail."
            )

        if mean and std and abs(mean) > 0 and std / abs(mean) > 1:
            notes.append(
                f'"{col}" has high variability — std dev ({std}) exceeds the mean ({mean}).'
            )

    return notes[:_AUTO_NOTES_LIMIT]


def generate_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, Any]:
    """
    Profile the most informative columns in the dataset.

    Limits profiling to the first `_MAX_CAT_PROFILE` categorical and first
    `_MAX_NUM_PROFILE` numeric columns to keep page load fast for wide datasets.

    Args:
        df:               Source DataFrame.
        numeric_cols:     Names of numeric columns.
        categorical_cols: Names of categorical columns.

    Returns:
        Dict with keys:
          - "top_values"       → {col: list of value/count/pct dicts}
          - "numeric_insights" → {col: stats dict}
          - "auto_notes"       → list of observation strings
    """
    cat_subset = categorical_cols[:_MAX_CAT_PROFILE]
    num_subset = numeric_cols[:_MAX_NUM_PROFILE]

    top_values       = {col: _top_values_for_column(df, col) for col in cat_subset}
    numeric_insights = {col: _profile_numeric_column(df, col) for col in num_subset}
    auto_notes       = _build_auto_notes(df, numeric_insights, top_values, categorical_cols)

    return {
        "top_values":       top_values,
        "numeric_insights": numeric_insights,
        "auto_notes":       auto_notes,
    }


# ── AI insights engine ─────────────────────────────────────────────────────────
# Each private function is a self-contained "insight generator" that returns a
# (possibly empty) list of InsightCard dicts.  generate_ai_insights() aggregates
# and deduplicates them.

def _trend_insights(df: pd.DataFrame, numeric_cols: list[str]) -> list[InsightCard]:
    """
    Detect rising or falling trends and frame them in business language.
    Reuses _detect_trend to avoid duplicating the half-mean comparison logic.
    """
    results: list[InsightCard] = []

    for col in numeric_cols[:5]:
        series = df[col].dropna()
        trend  = _detect_trend(series)

        if trend["direction"] == "flat":
            continue

        mid         = max(len(series) // 2, 1)
        first_mean  = float(series.iloc[:mid].mean())
        second_mean = float(series.iloc[mid:].mean())
        pct         = trend["change_pct"]

        if pct >= _SURGE_PCT:
            results.append(_make_insight(
                "positive", "📈",
                f"{col} surged {pct}% across the dataset",
                f"The average {col} jumped from {format_number(first_mean)} in the first half "
                f"to {format_number(second_mean)} in the second half — a strong upward signal.",
                f"+{pct}%",
            ))
        elif pct > 0:
            results.append(_make_insight(
                "positive", "↗",
                f"{col} grew steadily by {pct}%",
                f"First-half average: {format_number(first_mean)}. "
                f"Second-half average: {format_number(second_mean)}. "
                f"Consistent, moderate growth throughout the dataset.",
                f"+{pct}%",
            ))
        elif pct <= -_DROP_PCT:
            results.append(_make_insight(
                "negative", "📉",
                f"{col} dropped sharply by {abs(pct)}%",
                f"Average {col} fell from {format_number(first_mean)} to "
                f"{format_number(second_mean)} — a significant decline that may warrant investigation.",
                f"{pct}%",
            ))
        else:
            results.append(_make_insight(
                "warning", "↘",
                f"{col} declined by {abs(pct)}%",
                f"First-half average: {format_number(first_mean)}. "
                f"Second-half average: {format_number(second_mean)}. "
                f"A gradual downward movement is visible in the data.",
                f"{pct}%",
            ))

    return results


def _dominance_insights(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> list[InsightCard]:
    """
    Detect when one category heavily dominates a column.
    Flags concentration levels at ≥50%, ≥30%, or top-3 ≥80%.
    """
    results: list[InsightCard] = []
    total = len(df)

    for col in categorical_cols[:_MAX_CAT_PROFILE]:
        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            continue

        top_val  = counts.index[0]
        top_pct  = round(counts.iloc[0] / total * 100, 1)
        top3_pct = round(counts.iloc[:3].sum() / total * 100, 1) if len(counts) >= 3 else top_pct

        if top_pct >= 50:
            results.append(_make_insight(
                "info", "🏆",
                f'"{top_val}" dominates {col} with {top_pct}% share',
                f'More than half of all records in "{col}" belong to "{top_val}". '
                f"This level of concentration may indicate a key segment or a data imbalance.",
                f"{top_pct}%",
            ))
        elif top_pct >= 30:
            results.append(_make_insight(
                "info", "👑",
                f'"{top_val}" leads {col} at {top_pct}%',
                f'"{top_val}" is the clear front-runner in "{col}", representing nearly a third '
                f"of all entries. The top 3 values together cover {top3_pct}%.",
                f"{top_pct}%",
            ))
        elif len(counts) >= 3 and top3_pct >= 80:
            top3_labels = ", ".join(str(x) for x in counts.index[:3])
            results.append(_make_insight(
                "info", "📊",
                f"{col} is highly concentrated in 3 categories",
                f'The top 3 values in "{col}" — {top3_labels} — '
                f"together account for {top3_pct}% of all records.",
                f"{top3_pct}%",
            ))

    return results


def _correlation_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> list[InsightCard]:
    """
    Detect strong positive or negative correlations between numeric columns.
    Returns at most two findings to avoid overwhelming the user.
    """
    results: list[InsightCard] = []
    if len(numeric_cols) < 2:
        return results

    cols_subset = numeric_cols[:_CORR_MAX_COLS]
    try:
        corr = df[cols_subset].corr(numeric_only=True)
    except Exception:
        return results

    seen: set[tuple[str, str]] = set()

    for i, col_a in enumerate(cols_subset):
        for col_b in cols_subset[i + 1:]:
            pair = (min(col_a, col_b), max(col_a, col_b))
            if pair in seen:
                continue
            seen.add(pair)

            r = corr.loc[col_a, col_b]
            if pd.isna(r):
                continue
            r = round(float(r), 2)

            if r >= _CORR_VERY_STRONG:
                results.append(_make_insight(
                    "info", "🔗",
                    f"{col_a} and {col_b} move almost in lockstep (r={r})",
                    f"These two columns have a very strong positive correlation ({r}). "
                    f"When {col_a} rises, {col_b} tends to rise proportionally.",
                    f"r={r}",
                ))
            elif r >= _CORR_STRONG:
                results.append(_make_insight(
                    "info", "↔",
                    f"{col_a} and {col_b} are strongly correlated (r={r})",
                    f"A correlation of {r} suggests a meaningful relationship. "
                    f"Changes in {col_a} are likely linked to changes in {col_b}.",
                    f"r={r}",
                ))
            elif r <= -_CORR_STRONG:
                results.append(_make_insight(
                    "warning", "↕",
                    f"{col_a} and {col_b} move in opposite directions (r={r})",
                    f"A strong negative correlation ({r}) means as {col_a} increases, "
                    f"{col_b} tends to decrease — and vice versa.",
                    f"r={r}",
                ))

    return results[:2]


def _outlier_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> list[InsightCard]:
    """
    Flag columns containing a meaningful number of extreme outliers
    (values more than `_OUTLIER_SIGMA` standard deviations from the mean).
    Returns at most two findings.
    """
    results: list[InsightCard] = []
    total = len(df)

    for col in numeric_cols[:5]:
        series = df[col].dropna()
        if len(series) < _OUTLIER_MIN_ROWS:
            continue

        mean, std = series.mean(), series.std()
        if std == 0:
            continue

        n_outliers = int(((series - mean).abs() > _OUTLIER_SIGMA * std).sum())
        if n_outliers < 1:
            continue

        pct      = round(n_outliers / total * 100, 1)
        severity = "warning" if pct > 2 else "info"
        plural   = "s" if n_outliers > 1 else ""

        results.append(_make_insight(
            severity, "⚠️",
            f"{col} has {n_outliers} extreme outlier{plural}",
            f"{n_outliers} record{plural} ({pct}% of rows) in \"{col}\" fall more than "
            f"{_OUTLIER_SIGMA:.0f} standard deviations from the mean "
            f"({format_number(mean)}). These may skew averages and should be reviewed.",
            f"{n_outliers} rows",
        ))

    return results[:2]


def _group_comparison_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[InsightCard]:
    """
    Compare group means: which category drives the highest/lowest average
    for the primary numeric column?  Returns at most two findings.
    """
    results: list[InsightCard] = []
    if not numeric_cols or not categorical_cols:
        return results

    num_col = numeric_cols[0]

    for cat_col in categorical_cols[:2]:
        try:
            n_unique = df[cat_col].nunique()
            if not (2 <= n_unique <= 30):
                continue

            group_means = df.groupby(cat_col)[num_col].mean().dropna()
            if len(group_means) < 2:
                continue

            best_cat  = group_means.idxmax()
            worst_cat = group_means.idxmin()
            best_v    = round(float(group_means.max()), 2)
            worst_v   = round(float(group_means.min()), 2)

            if worst_v == 0:
                continue

            diff_pct = round((best_v - worst_v) / abs(worst_v) * 100, 1)
            if diff_pct > 10:
                results.append(_make_insight(
                    "positive", "🥇",
                    f'"{best_cat}" leads in {num_col} with avg {format_number(best_v)}',
                    f'Across {cat_col} groups, "{best_cat}" has the highest average '
                    f"{num_col} ({format_number(best_v)}), while \"{worst_cat}\" has "
                    f"the lowest ({format_number(worst_v)}) — a {diff_pct}% gap.",
                    f"{diff_pct}% gap",
                ))
        except Exception:
            continue

    return results[:2]


def _distribution_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
) -> list[InsightCard]:
    """
    Detect heavily skewed distributions and significant mean/median divergence.
    Returns at most two findings.
    """
    results: list[InsightCard] = []

    for col in numeric_cols[:5]:
        series = df[col].dropna()
        if series.empty:
            continue

        skew   = round(float(series.skew()), 2)
        mean   = series.mean()
        median = series.median()

        if abs(skew) > _SKEW_HEAVY:
            tail = "upper tail (right-skewed)" if skew > 0 else "lower tail (left-skewed)"
            results.append(_make_insight(
                "warning", "〰",
                f"{col} has a heavy {tail} (skew={skew})",
                f"The distribution of {col} is not symmetric. Most values cluster low, "
                f"but extreme values pull the mean ({format_number(mean)}) away from the "
                f"median ({format_number(median)}). Median may be a better central measure here.",
                f"skew={skew}",
            ))
        elif median != 0 and abs(mean - median) / abs(median) > _MEAN_MEDIAN_DIVERGENCE:
            direction = "above" if mean > median else "below"
            results.append(_make_insight(
                "info", "📐",
                f"Mean and median diverge significantly for {col}",
                f"The mean ({format_number(mean)}) is {direction} the median "
                f"({format_number(median)}) by more than "
                f"{int(_MEAN_MEDIAN_DIVERGENCE * 100)}%, suggesting outliers or a "
                f"skewed distribution are influencing the average.",
                f"Δ={format_number(abs(mean - median))}",
            ))

    return results[:2]


def _data_quality_insights(
    df: pd.DataFrame,
) -> list[InsightCard]:
    """
    Surface columns with >10% missing data so users know where to focus
    data-cleaning efforts.  Returns at most two findings.
    """
    results: list[InsightCard] = []
    total = len(df)
    null_counts = df.isnull().sum()

    high_null = [
        (col, int(n))
        for col, n in null_counts.items()
        if n / total > 0.10
    ]
    high_null.sort(key=lambda x: -x[1])

    for col, n in high_null[:2]:
        pct = round(n / total * 100, 1)
        results.append(_make_insight(
            "warning", "🕳",
            f"{col} is missing {pct}% of values",
            f"{n:,} out of {total:,} records have no value for \"{col}\". "
            f"This may affect analysis accuracy and should be addressed before "
            f"drawing conclusions.",
            f"{pct}% null",
        ))

    return results


def _completeness_insight(df: pd.DataFrame) -> InsightCard | None:
    """
    Return a single overall data-quality card (positive or informational).
    Returns None when completeness is below 90% (covered by _data_quality_insights).
    """
    total_cells = df.size
    missing     = int(df.isnull().sum().sum())
    pct_complete = round((total_cells - missing) / total_cells * 100, 1)

    if pct_complete >= 99:
        return _make_insight(
            "positive", "✅",
            f"Dataset is {pct_complete}% complete — excellent data quality",
            f"Only {missing:,} missing values across {total_cells:,} total cells. "
            f"This dataset is well-suited for reliable analysis.",
            f"{pct_complete}%",
        )
    if pct_complete >= 90:
        return _make_insight(
            "info", "📋",
            f"Dataset completeness is {pct_complete}%",
            f"{missing:,} missing values detected across {total_cells:,} cells. "
            f"Mostly clean — minor gaps may affect column-level insights.",
            f"{pct_complete}%",
        )
    return None


def generate_ai_insights(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> list[InsightCard]:
    """
    Run all rule-based insight generators and return a deduplicated list
    of at most `_MAX_AI_INSIGHTS` insight cards.

    Each card has keys: severity, icon, headline, body, metric.
    Severity levels: "positive" | "negative" | "warning" | "info".

    Args:
        df:               Source DataFrame.
        numeric_cols:     Names of numeric columns.
        categorical_cols: Names of categorical columns.

    Returns:
        List of InsightCard dicts, deduplicated by headline.
    """
    all_insights: list[InsightCard] = []

    # Collect from all generators
    all_insights.extend(_trend_insights(df, numeric_cols))
    all_insights.extend(_dominance_insights(df, categorical_cols))
    all_insights.extend(_group_comparison_insights(df, numeric_cols, categorical_cols))
    all_insights.extend(_correlation_insights(df, numeric_cols))
    all_insights.extend(_distribution_insights(df, numeric_cols))
    all_insights.extend(_outlier_insights(df, numeric_cols))
    all_insights.extend(_data_quality_insights(df))

    quality_card = _completeness_insight(df)
    if quality_card:
        all_insights.append(quality_card)

    # Deduplicate by headline (preserves insertion order)
    seen: set[str] = set()
    unique: list[InsightCard] = []
    for card in all_insights:
        if card["headline"] not in seen:
            seen.add(card["headline"])
            unique.append(card)

    return unique[:_MAX_AI_INSIGHTS]
