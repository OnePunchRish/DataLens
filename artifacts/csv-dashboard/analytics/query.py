"""
analytics/query.py — Natural-language query engine.

Translates plain-English questions into pandas operations over a loaded
DataFrame and returns formatted text answers.

Supported query categories:
  - Dataset overview  (rows, columns, column names, missing values, summary stats)
  - Aggregates        (average, sum, max, min, median, std dev)
  - Distribution      (top N values, unique / distinct counts)
  - Relationships     (correlation between columns)
  - Group analysis    (group-by aggregation)
  - Column profile    (describe / show a specific column)
  - Count queries     (value counts, non-null counts)
"""

from __future__ import annotations

import re

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

# Number of columns shown in multi-column aggregate responses
_AGG_DISPLAY_LIMIT = 6

# Columns included in the all-pairs correlation table
_CORR_COLS_LIMIT = 5

# Default and maximum rows for top-N queries when not specified
_TOP_N_DEFAULT = 5

# Max categories returned in group-by responses
_GROUP_BY_LIMIT = 8

# Max value-count entries shown for categorical columns
_COUNT_CAT_LIMIT = 8


# ── Internal helpers ───────────────────────────────────────────────────────────

def _extract_column_refs(query: str, all_cols: list[str]) -> list[str]:
    """
    Return column names that appear verbatim in `query` (case-insensitive),
    sorted longest-match first to prefer specific names over substrings.

    Args:
        query:    Lowercased user query string.
        all_cols: Full list of column names in the DataFrame.

    Returns:
        Ordered list of matched column names.
    """
    q_lower = query.lower()
    return [
        col
        for col in sorted(all_cols, key=len, reverse=True)
        if col.lower() in q_lower
    ]


def _keyword_match(query: str, *keywords: str) -> bool:
    """
    Return True if any keyword from `keywords` is found as a substring
    of the lowercased `query`.

    Args:
        query:    Lowercased user query string.
        *keywords: One or more keyword strings to search for.
    """
    return any(kw in query for kw in keywords)


def _agg_lines(
    df: pd.DataFrame,
    cols: list[str],
    agg_fn,
    round_dp: int = 2,
) -> str:
    """
    Compute an aggregate (mean, sum, etc.) for each column and return a
    bullet-list string ready for display.

    Args:
        df:       Source DataFrame.
        cols:     Numeric column names to aggregate.
        agg_fn:   Callable that accepts a Series and returns a scalar
                  (e.g. pd.Series.mean).
        round_dp: Decimal places for rounding.

    Returns:
        Multi-line bullet string, e.g. "  • price: 42.5"
    """
    lines = [f"  • {c}: {round(float(agg_fn(df[c])), round_dp):,}" for c in cols]
    return "\n".join(lines)


# ── Query handlers ─────────────────────────────────────────────────────────────
# Each handler returns a string response or None if it does not apply.

def _handle_overview(
    query: str,
    df: pd.DataFrame,
    all_cols: list[str],
    num_cols: list[str],
    rows: int,
) -> str | None:
    """Handle dataset-level overview questions (row count, column list, etc.)."""

    if _keyword_match(query, "how many rows", "row count", "how many records",
                      "total rows", "number of rows"):
        return f"The dataset has {rows:,} rows."

    if _keyword_match(query, "how many columns", "column count",
                      "number of columns", "total columns"):
        return f"The dataset has {len(all_cols)} columns."

    if _keyword_match(query, "show columns", "list columns", "what columns",
                      "all columns", "column names", "what are the columns"):
        bullet_list = "\n".join(f"  • {c}" for c in all_cols)
        return f"Columns in this dataset:\n{bullet_list}"

    if _keyword_match(query, "missing", "null", "empty", "nan"):
        null_counts = df.isnull().sum()
        missing = [(c, int(n)) for c, n in null_counts.items() if n > 0]
        if not missing:
            return "No missing values — this dataset is fully complete."
        lines = [f"  • {c}: {n:,} missing ({n / rows * 100:.1f}%)" for c, n in missing]
        return f"Missing values found in {len(missing)} column(s):\n" + "\n".join(lines)

    if _keyword_match(query, "describe", "summary", "statistics", "stats overview"):
        if not num_cols:
            return "No numeric columns found."
        desc = df[num_cols].describe().round(2)
        lines = [
            f"  • {c}: mean={desc.loc['mean', c]}, "
            f"min={desc.loc['min', c]}, max={desc.loc['max', c]}"
            for c in num_cols[:_AGG_DISPLAY_LIMIT]
        ]
        return "Numeric column summary:\n" + "\n".join(lines)

    return None


def _handle_aggregates(
    query: str,
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    target: str | None,
    rows: int,
) -> str | None:
    """Handle aggregate queries: average, sum, max, min, median, std dev."""

    # Average / mean
    if _keyword_match(query, "average", "mean", "avg"):
        if target and target in num_cols:
            v = round(float(df[target].mean()), 3)
            return f"The average of '{target}' is {v:,}."
        if num_cols:
            return "Average of numeric columns:\n" + _agg_lines(df, num_cols[:_AGG_DISPLAY_LIMIT], pd.Series.mean)
        return "No numeric columns available."

    # Sum / total
    if _keyword_match(query, "sum of", "total of", "sum ", "total "):
        if target and target in num_cols:
            v = round(float(df[target].sum()), 2)
            return f"The total sum of '{target}' is {v:,}."
        if num_cols:
            return "Sum of numeric columns:\n" + _agg_lines(df, num_cols[:_AGG_DISPLAY_LIMIT], pd.Series.sum)
        return "No numeric columns available."

    # Max / highest
    if _keyword_match(query, "max", "maximum", "highest", "largest", "biggest"):
        if target and target in num_cols:
            return f"The maximum value in '{target}' is {df[target].max():,}."
        if target and target in cat_cols:
            top = df[target].value_counts()
            return f"The most common value in '{target}' is '{top.index[0]}' ({int(top.iloc[0]):,} times)."
        if num_cols:
            maxes = {c: float(df[c].max()) for c in num_cols}
            best  = max(maxes, key=maxes.get)
            lines = "\n".join(
                f"  • {c}: {v:,}"
                for c, v in sorted(maxes.items(), key=lambda x: -x[1])[:_AGG_DISPLAY_LIMIT]
            )
            return f"'{best}' has the highest maximum value: {maxes[best]:,}.\n{lines}"
        return "No numeric columns available."

    # Min / lowest
    if _keyword_match(query, "min", "minimum", "lowest", "smallest"):
        if target and target in num_cols:
            return f"The minimum value in '{target}' is {df[target].min():,}."
        if num_cols:
            mins = {c: float(df[c].min()) for c in num_cols}
            best = min(mins, key=mins.get)
            return f"'{best}' has the lowest minimum value: {mins[best]:,}."
        return "No numeric columns available."

    # Median
    if _keyword_match(query, "median"):
        if target and target in num_cols:
            v = round(float(df[target].median()), 3)
            return f"The median of '{target}' is {v:,}."
        if num_cols:
            return "Median of numeric columns:\n" + _agg_lines(df, num_cols[:_AGG_DISPLAY_LIMIT], pd.Series.median)
        return "No numeric columns available."

    # Standard deviation
    if _keyword_match(query, "std", "standard deviation", "variance", "spread"):
        if target and target in num_cols:
            v = round(float(df[target].std()), 3)
            return f"The standard deviation of '{target}' is {v:,}."
        if num_cols:
            return "Standard deviation of numeric columns:\n" + _agg_lines(df, num_cols[:_AGG_DISPLAY_LIMIT], pd.Series.std)
        return "No numeric columns available."

    return None


def _handle_distribution(
    query: str,
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    target: str | None,
    rows: int,
) -> str | None:
    """Handle top-N value counts and unique/distinct count queries."""

    # Top N / most common
    if _keyword_match(query, "top", "most common", "frequent", "popular",
                      "distribution of", "value counts"):
        col = target or (cat_cols[0] if cat_cols else num_cols[0] if num_cols else None)
        if col is None:
            return "No columns available for value counts."

        match = re.search(r"\btop\s+(\d+)\b", query)
        n      = int(match.group(1)) if match else _TOP_N_DEFAULT
        counts = df[col].value_counts().head(n)
        lines  = [
            f"  #{i + 1}  '{v}' — {c:,} ({c / rows * 100:.1f}%)"
            for i, (v, c) in enumerate(counts.items())
        ]
        return f"Top {n} values in '{col}':\n" + "\n".join(lines)

    # Unique / distinct
    if _keyword_match(query, "unique", "distinct", "how many different"):
        if target:
            return f"'{target}' has {df[target].nunique():,} unique values."
        lines = [f"  • {c}: {df[c].nunique():,}" for c in df.columns]
        return "Unique value counts per column:\n" + "\n".join(lines)

    return None


def _handle_correlation(
    query: str,
    df: pd.DataFrame,
    num_cols: list[str],
    mentioned: list[str],
) -> str | None:
    """Handle correlation queries between pairs of numeric columns."""

    if not _keyword_match(query, "correlation", "correlated", "relationship between", "related to"):
        return None

    # Two specific columns mentioned
    num_mentioned = [c for c in mentioned if c in num_cols]
    if len(num_mentioned) >= 2:
        col_a, col_b = num_mentioned[0], num_mentioned[1]
        r = round(float(df[[col_a, col_b]].corr().loc[col_a, col_b]), 3)
        strength  = (
            "very strong" if abs(r) >= 0.8 else
            "strong"      if abs(r) >= 0.6 else
            "moderate"    if abs(r) >= 0.4 else
            "weak"
        )
        direction = "positive" if r > 0 else "negative"
        return (
            f"Correlation between '{col_a}' and '{col_b}': r = {r}\n"
            f"That's a {strength} {direction} relationship."
        )

    # All-pairs summary
    if len(num_cols) >= 2:
        subset = num_cols[:_CORR_COLS_LIMIT]
        corr   = df[subset].corr().round(2)
        pairs  = sorted(
            [(a, b, float(corr.loc[a, b]))
             for i, a in enumerate(subset)
             for b in subset[i + 1:]],
            key=lambda x: -abs(x[2]),
        )
        lines = [f"  • '{a}' & '{b}': r={r}" for a, b, r in pairs[:5]]
        return "Strongest correlations:\n" + "\n".join(lines)

    return "Need at least 2 numeric columns for correlation."


def _handle_group_by(
    query: str,
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    mentioned: list[str],
) -> str | None:
    """Handle group-by aggregation queries (average or sum per category)."""

    if not _keyword_match(query, "group by", "grouped by", "per ", "by each",
                          "average per", "mean per", "sum per"):
        return None

    # Try to find a (categorical, numeric) pair from the query
    mentioned_cats = [c for c in mentioned if c in cat_cols]
    mentioned_nums = [c for c in mentioned if c in num_cols]
    pairs = [(ca, cn) for ca in mentioned_cats for cn in mentioned_nums]

    if pairs:
        cat_col, num_col = pairs[0]
    elif cat_cols and num_cols:
        cat_col, num_col = cat_cols[0], num_cols[0]
    else:
        return None

    use_sum = _keyword_match(query, "sum", "total")
    agg_fn  = pd.Series.sum if use_sum else pd.Series.mean
    label   = "Total" if use_sum else "Average"

    grouped = (
        df.groupby(cat_col)[num_col]
        .agg(agg_fn)
        .round(2)
        .sort_values(ascending=False)
        .head(_GROUP_BY_LIMIT)
    )
    lines = [f"  • {k}: {v:,}" for k, v in grouped.items()]
    return f"{label} '{num_col}' by '{cat_col}':\n" + "\n".join(lines)


def _handle_column_profile(
    query: str,
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    target: str | None,
    rows: int,
) -> str | None:
    """Handle 'show / describe / tell me about' queries for a specific column."""

    if not (_keyword_match(query, "show", "tell me about", "describe", "info about") and target):
        return None

    if target in num_cols:
        s = df[target].describe().round(2)
        return (
            f"'{target}' statistics:\n"
            f"  • Count:   {int(s['count']):,}\n"
            f"  • Mean:    {s['mean']:,}\n"
            f"  • Std Dev: {s['std']:,}\n"
            f"  • Min:     {s['min']:,}\n"
            f"  • Median:  {s['50%']:,}\n"
            f"  • Max:     {s['max']:,}"
        )

    if target in cat_cols:
        counts   = df[target].value_counts().head(5)
        n_unique = df[target].nunique()
        lines    = [
            f"  • '{v}': {c:,} ({c / rows * 100:.1f}%)"
            for v, c in counts.items()
        ]
        return f"'{target}' — {n_unique:,} unique values. Top 5:\n" + "\n".join(lines)

    return None


def _handle_count(
    query: str,
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    target: str | None,
    rows: int,
) -> str | None:
    """Handle count / how-many queries for a specific column."""

    if not (_keyword_match(query, "count", "how many") and target):
        return None

    if target in cat_cols:
        counts = df[target].value_counts().head(_COUNT_CAT_LIMIT)
        lines  = [f"  • '{v}': {c:,}" for v, c in counts.items()]
        return f"Value counts for '{target}':\n" + "\n".join(lines)

    if target in num_cols:
        n = int(df[target].count())
        return f"'{target}' has {n:,} non-null values out of {rows:,} rows."

    return None


def _fallback_response(
    num_cols: list[str],
    cat_cols: list[str],
) -> str:
    """
    Return a helpful fallback message with dataset-specific example queries
    when no handler matches the user's input.
    """
    examples: list[str] = []
    if num_cols:
        examples.append(f"average of {num_cols[0]}")
        examples.append(f"max {num_cols[0]}")
    if cat_cols:
        examples.append(f"top 5 {cat_cols[0]}")
    if len(num_cols) >= 2:
        examples.append(f"correlation between {num_cols[0]} and {num_cols[1]}")
    if cat_cols and num_cols:
        examples.append(f"average {num_cols[0]} by {cat_cols[0]}")

    example_lines = "\n".join(f'  • "{e}"' for e in examples)
    return (
        "I could not match that query. Try asking:\n"
        f"{example_lines}\n"
        '  • "how many rows"\n'
        '  • "missing data"\n'
        '  • "show columns"'
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def answer_question(df: pd.DataFrame, question: str) -> str:
    """
    Interpret a plain-English question about `df` and return a formatted answer.

    Dispatches to one of eight handler functions in priority order.
    Returns a helpful fallback with example queries if no handler matches.

    Args:
        df:       The loaded dataset as a pandas DataFrame.
        question: Raw user question string.

    Returns:
        A plain-text answer, possibly multi-line with bullet points.
    """
    query = question.lower().strip()

    all_cols  = df.columns.tolist()
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    cat_cols  = df.select_dtypes(include=["object", "category"]).columns.tolist()
    rows      = len(df)
    mentioned = _extract_column_refs(query, all_cols)
    target    = mentioned[0] if mentioned else None

    handlers = [
        _handle_overview(query, df, all_cols, num_cols, rows),
        _handle_aggregates(query, df, num_cols, cat_cols, target, rows),
        _handle_distribution(query, df, num_cols, cat_cols, target, rows),
        _handle_correlation(query, df, num_cols, mentioned),
        _handle_group_by(query, df, num_cols, cat_cols, mentioned),
        _handle_column_profile(query, df, num_cols, cat_cols, target, rows),
        _handle_count(query, df, num_cols, cat_cols, target, rows),
    ]

    for response in handlers:
        if response is not None:
            return response

    return _fallback_response(num_cols, cat_cols)
