import os
import json
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"csv"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ── helpers ───────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataframe():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None


def fmt(n):
    """Format a number nicely for display."""
    if isinstance(n, float):
        if abs(n) >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        if abs(n) >= 1_000:
            return f"{n / 1_000:.1f}K"
        return f"{n:,.2f}"
    return f"{n:,}"


# ── column-level insights engine ─────────────────────────────────────────────

def detect_trend(series: pd.Series) -> dict:
    clean = series.dropna()
    if len(clean) < 10:
        return {"direction": "flat", "change_pct": 0.0}
    mid = len(clean) // 2
    first_mean = clean.iloc[:mid].mean()
    second_mean = clean.iloc[mid:].mean()
    if first_mean == 0:
        return {"direction": "flat", "change_pct": 0.0}
    change_pct = round((second_mean - first_mean) / abs(first_mean) * 100, 1)
    direction = "up" if change_pct > 5 else ("down" if change_pct < -5 else "flat")
    return {"direction": direction, "change_pct": change_pct}


def numeric_column_insights(df: pd.DataFrame, col: str) -> dict:
    s = df[col].dropna()
    if s.empty:
        return {}
    return {
        "mean":     round(float(s.mean()), 3),
        "median":   round(float(s.median()), 3),
        "std":      round(float(s.std()), 3),
        "min":      round(float(s.min()), 3),
        "max":      round(float(s.max()), 3),
        "range":    round(float(s.max() - s.min()), 3),
        "skew":     round(float(s.skew()), 2),
        "trend":    detect_trend(s),
        "non_null": int(s.count()),
    }


def top_values_for_column(df: pd.DataFrame, col: str, n: int = 5) -> list:
    total = len(df)
    counts = df[col].value_counts(dropna=True).head(n)
    return [{"value": str(v), "count": int(c), "pct": round(c / total * 100, 1)}
            for v, c in counts.items()]


def generate_auto_notes(df, numeric_cols, categorical_cols, numeric_insights, top_values):
    notes = []
    rows = len(df)

    if categorical_cols:
        unique_counts = {c: df[c].nunique() for c in categorical_cols}
        most_varied = max(unique_counts, key=unique_counts.get)
        notes.append(
            f'"{most_varied}" has the most variety -- '
            f'{unique_counts[most_varied]:,} unique values across {rows:,} rows.'
        )

    if top_values:
        first_cat = next(iter(top_values))
        top = top_values[first_cat]
        if top:
            notes.append(
                f'The most common "{first_cat}" is '
                f'"{top[0]["value"]}" ({top[0]["pct"]}% of rows).'
            )

    for col, info in numeric_insights.items():
        trend = info.get("trend", {})
        if trend.get("direction") == "up":
            notes.append(f'"{col}" shows an upward trend (+{trend["change_pct"]}% from first half to second half).')
        elif trend.get("direction") == "down":
            notes.append(f'"{col}" shows a downward trend ({trend["change_pct"]}% from first half to second half).')

    for col, info in numeric_insights.items():
        skew = info.get("skew", 0)
        if abs(skew) > 1.5:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            notes.append(f'"{col}" is heavily skewed {direction} (skewness = {skew}), suggesting outliers or a long tail.')

    for col, info in numeric_insights.items():
        mean = info.get("mean", 0)
        std = info.get("std", 0)
        if mean and std and std / abs(mean) > 1:
            notes.append(f'"{col}" has high variability -- std dev ({std}) exceeds the mean ({mean}).')

    return notes[:6]


def generate_insights(df, numeric_cols, categorical_cols):
    featured_cat = categorical_cols[:4]
    featured_num = numeric_cols[:6]
    top_values       = {col: top_values_for_column(df, col) for col in featured_cat}
    numeric_insights = {col: numeric_column_insights(df, col) for col in featured_num}
    auto_notes       = generate_auto_notes(df, numeric_cols, categorical_cols,
                                           numeric_insights, top_values)
    return {
        "top_values":       top_values,
        "numeric_insights": numeric_insights,
        "auto_notes":       auto_notes,
    }


# ── AI insights engine ────────────────────────────────────────────────────────
# Rules simulate analyst-grade reasoning. No external API required.

def _insight(severity, icon, headline, body, metric=None):
    return {"severity": severity, "icon": icon,
            "headline": headline, "body": body, "metric": metric}


def _trend_insights(df, numeric_cols):
    """Detect rising or falling trends and frame them in business language."""
    results = []
    for col in numeric_cols[:5]:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        mid = len(s) // 2
        f_mean = s.iloc[:mid].mean()
        s_mean = s.iloc[mid:].mean()
        if f_mean == 0:
            continue
        pct = round((s_mean - f_mean) / abs(f_mean) * 100, 1)

        if pct >= 20:
            results.append(_insight(
                "positive", "📈",
                f"{col} surged {pct}% across the dataset",
                f"The average {col} jumped from {fmt(f_mean)} in the first half to "
                f"{fmt(s_mean)} in the second half -- a strong upward signal.",
                f"+{pct}%"
            ))
        elif pct >= 5:
            results.append(_insight(
                "positive", "↗",
                f"{col} grew steadily by {pct}%",
                f"First-half average: {fmt(f_mean)}. Second-half average: {fmt(s_mean)}. "
                f"Consistent, moderate growth throughout the dataset.",
                f"+{pct}%"
            ))
        elif pct <= -20:
            results.append(_insight(
                "negative", "📉",
                f"{col} dropped sharply by {abs(pct)}%",
                f"Average {col} fell from {fmt(f_mean)} to {fmt(s_mean)} -- "
                f"a significant decline that may warrant investigation.",
                f"{pct}%"
            ))
        elif pct <= -5:
            results.append(_insight(
                "warning", "↘",
                f"{col} declined by {abs(pct)}%",
                f"First-half average: {fmt(f_mean)}. Second-half average: {fmt(s_mean)}. "
                f"A gradual downward movement is visible in the data.",
                f"{pct}%"
            ))
    return results


def _dominance_insights(df, categorical_cols):
    """Detect when one category heavily dominates a column."""
    results = []
    total = len(df)
    for col in categorical_cols[:4]:
        counts = df[col].value_counts(dropna=True)
        if counts.empty:
            continue
        top_val   = counts.index[0]
        top_pct   = round(counts.iloc[0] / total * 100, 1)
        top3_pct  = round(counts.iloc[:3].sum() / total * 100, 1) if len(counts) >= 3 else top_pct

        if top_pct >= 50:
            results.append(_insight(
                "info", "🏆",
                f'"{top_val}" dominates {col} with {top_pct}% share',
                f'More than half of all records in "{col}" belong to "{top_val}". '
                f"This level of concentration may indicate a key segment or a data imbalance.",
                f"{top_pct}%"
            ))
        elif top_pct >= 30:
            results.append(_insight(
                "info", "👑",
                f'"{top_val}" leads {col} at {top_pct}%',
                f'"{top_val}" is the clear front-runner in "{col}", representing nearly a third of all entries. '
                f"The top 3 values together cover {top3_pct}%.",
                f"{top_pct}%"
            ))
        elif len(counts) >= 3 and top3_pct >= 80:
            top3_labels = ', '.join(str(x) for x in counts.index[:3])
            results.append(_insight(
                "info", "📊",
                f"{col} is highly concentrated in 3 categories",
                f'The top 3 values in "{col}" -- ' + top3_labels +
                f' -- together account for {top3_pct}% of all records.',
                f"{top3_pct}%"
            ))
    return results


def _correlation_insights(df, numeric_cols):
    """Detect strong positive or negative correlations between numeric columns."""
    results = []
    if len(numeric_cols) < 2:
        return results
    try:
        corr = df[numeric_cols].corr(numeric_only=True)
    except Exception:
        return results

    seen = set()
    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1:]:
            pair = tuple(sorted([col_a, col_b]))
            if pair in seen:
                continue
            seen.add(pair)
            r = corr.loc[col_a, col_b]
            if pd.isna(r):
                continue
            r = round(float(r), 2)

            if r >= 0.85:
                results.append(_insight(
                    "info", "🔗",
                    f"{col_a} and {col_b} move almost in lockstep (r={r})",
                    f"These two columns have a very strong positive correlation ({r}). "
                    f"When {col_a} rises, {col_b} tends to rise proportionally.",
                    f"r={r}"
                ))
            elif r >= 0.7:
                results.append(_insight(
                    "info", "↔",
                    f"{col_a} and {col_b} are strongly correlated (r={r})",
                    f"A correlation of {r} suggests a meaningful relationship. "
                    f"Changes in {col_a} are likely linked to changes in {col_b}.",
                    f"r={r}"
                ))
            elif r <= -0.7:
                results.append(_insight(
                    "warning", "↕",
                    f"{col_a} and {col_b} move in opposite directions (r={r})",
                    f"A strong negative correlation ({r}) means as {col_a} increases, "
                    f"{col_b} tends to decrease -- and vice versa.",
                    f"r={r}"
                ))
    return results[:2]  # keep concise


def _outlier_insights(df, numeric_cols):
    """Flag columns with a meaningful number of extreme outliers (>3σ)."""
    results = []
    total = len(df)
    for col in numeric_cols[:5]:
        s = df[col].dropna()
        if len(s) < 20:
            continue
        mean, std = s.mean(), s.std()
        if std == 0:
            continue
        n_outliers = int(((s - mean).abs() > 3 * std).sum())
        pct = round(n_outliers / total * 100, 1)
        if n_outliers >= 1:
            severity = "warning" if pct > 2 else "info"
            results.append(_insight(
                severity, "⚠️",
                f"{col} has {n_outliers} extreme outlier{'s' if n_outliers > 1 else ''}",
                f"{n_outliers} record{'s' if n_outliers > 1 else ''} ({pct}% of rows) in \"{col}\" "
                f"fall more than 3 standard deviations from the mean ({fmt(mean)}). "
                f"These may skew averages and should be reviewed.",
                f"{n_outliers} rows"
            ))
    return results[:2]


def _group_comparison_insights(df, numeric_cols, categorical_cols):
    """Compare group means: which category has the highest/lowest average?"""
    results = []
    if not numeric_cols or not categorical_cols:
        return results

    num_col = numeric_cols[0]
    for cat_col in categorical_cols[:2]:
        try:
            n_unique = df[cat_col].nunique()
            if n_unique < 2 or n_unique > 30:
                continue
            group_means = df.groupby(cat_col)[num_col].mean().dropna()
            if group_means.empty or len(group_means) < 2:
                continue
            best    = group_means.idxmax()
            worst   = group_means.idxmin()
            best_v  = round(float(group_means.max()), 2)
            worst_v = round(float(group_means.min()), 2)
            diff_pct = round((best_v - worst_v) / abs(worst_v) * 100, 1) if worst_v != 0 else 0
            if diff_pct > 10:
                results.append(_insight(
                    "positive", "🥇",
                    f'"{best}" leads in {num_col} with avg {fmt(best_v)}',
                    f'Across {cat_col} groups, "{best}" has the highest average {num_col} ({fmt(best_v)}), '
                    f'while "{worst}" has the lowest ({fmt(worst_v)}) -- a {diff_pct}% gap.',
                    f"{diff_pct}% gap"
                ))
        except Exception:
            continue
    return results[:2]


def _distribution_insights(df, numeric_cols):
    """Detect skewed distributions and mean/median divergence."""
    results = []
    for col in numeric_cols[:5]:
        s = df[col].dropna()
        if s.empty:
            continue
        skew = round(float(s.skew()), 2)
        mean   = s.mean()
        median = s.median()

        if abs(skew) > 2:
            tail = "upper tail (right-skewed)" if skew > 0 else "lower tail (left-skewed)"
            results.append(_insight(
                "warning", "〰",
                f"{col} has a heavy {tail} (skew={skew})",
                f"The distribution of {col} is not symmetric. Most values cluster low, "
                f"but a few high values pull the mean ({fmt(mean)}) well above the median ({fmt(median)}). "
                f"Median may be a better central measure here.",
                f"skew={skew}"
            ))

        elif median != 0 and abs(mean - median) / abs(median) > 0.3:
            direction = "above" if mean > median else "below"
            results.append(_insight(
                "info", "📐",
                f"Mean and median diverge significantly for {col}",
                f"The mean ({fmt(mean)}) is {direction} the median ({fmt(median)}) by more than 30%, "
                f"suggesting outliers or a skewed distribution are influencing the average.",
                f"Δ={fmt(abs(mean-median))}"
            ))
    return results[:2]


def _data_quality_insights(df, numeric_cols, categorical_cols):
    """Surface columns with notable missing data."""
    results = []
    total = len(df)
    null_counts = df.isnull().sum()
    high_null = [(col, int(n)) for col, n in null_counts.items() if n / total > 0.1]
    high_null.sort(key=lambda x: -x[1])
    for col, n in high_null[:2]:
        pct = round(n / total * 100, 1)
        results.append(_insight(
            "warning", "🕳",
            f"{col} is missing {pct}% of values",
            f"{n:,} out of {total:,} records have no value for \"{col}\". "
            f"This may affect analysis accuracy and should be addressed before drawing conclusions.",
            f"{pct}% null"
        ))
    return results


def _completeness_insight(df):
    """Overall completeness score for the dataset."""
    total_cells = df.size
    missing     = int(df.isnull().sum().sum())
    pct_complete = round((total_cells - missing) / total_cells * 100, 1)
    if pct_complete >= 99:
        return _insight(
            "positive", "✅",
            f"Dataset is {pct_complete}% complete -- excellent data quality",
            f"Only {missing:,} missing values across {total_cells:,} total cells. "
            f"This dataset is well-suited for reliable analysis.",
            f"{pct_complete}%"
        )
    elif pct_complete >= 90:
        return _insight(
            "info", "📋",
            f"Dataset completeness is {pct_complete}%",
            f"{missing:,} missing values detected across {total_cells:,} cells. "
            f"Mostly clean -- minor gaps may affect column-level insights.",
            f"{pct_complete}%"
        )
    return None  # only return positive/info for quality banner


def generate_ai_insights(df, numeric_cols, categorical_cols):
    """
    Rule-based AI insights engine.
    Produces structured insight objects with severity, icon, headline, body, metric.
    No external APIs required -- all logic is derived from data patterns.
    """
    bucket = []

    # Run all insight generators
    bucket.extend(_trend_insights(df, numeric_cols))
    bucket.extend(_dominance_insights(df, categorical_cols))
    bucket.extend(_group_comparison_insights(df, numeric_cols, categorical_cols))
    bucket.extend(_correlation_insights(df, numeric_cols))
    bucket.extend(_distribution_insights(df, numeric_cols))
    bucket.extend(_outlier_insights(df, numeric_cols))
    bucket.extend(_data_quality_insights(df, numeric_cols, categorical_cols))

    quality = _completeness_insight(df)
    if quality:
        bucket.append(quality)

    # Deduplicate and cap
    seen = set()
    unique = []
    for item in bucket:
        key = item["headline"]
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return unique[:9]  # show at most 9 insights


# ── chart builder ─────────────────────────────────────────────────────────────

CHART_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    title_font_size=16,
    margin=dict(l=40, r=20, t=50, b=60),
)
AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.08)")


def build_charts(df, numeric_cols, categorical_cols):
    charts = {}

    if numeric_cols and categorical_cols:
        bar_x = categorical_cols[0]
        bar_y = numeric_cols[0]
        top_vals = df[bar_x].value_counts().head(15).index
        bar_df = df[df[bar_x].isin(top_vals)]
        fig = px.bar(bar_df, x=bar_x, y=bar_y,
                     title=f"{bar_y} by {bar_x}", color=bar_x,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**CHART_THEME, showlegend=False)
        fig.update_xaxes(**AXIS_STYLE, tickfont_size=11)
        fig.update_yaxes(**AXIS_STYLE)
        charts["bar"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    if categorical_cols:
        pie_col = categorical_cols[0]
        counts = df[pie_col].value_counts().head(10)
        fig = px.pie(values=counts.values, names=counts.index,
                     title=f"Distribution of {pie_col}",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(**{**CHART_THEME, "margin": dict(l=20, r=20, t=50, b=20)})
        fig.update_traces(textfont_color="#1a1a2e")
        charts["pie"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    if len(numeric_cols) >= 2:
        line_x, line_y = numeric_cols[0], numeric_cols[1]
        line_df = df[[line_x, line_y]].dropna().sort_values(line_x).head(200)
        fig = px.line(line_df, x=line_x, y=line_y,
                      title=f"{line_y} over {line_x}",
                      color_discrete_sequence=["#6366f1"])
        fig.update_layout(**CHART_THEME)
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        charts["line"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    elif len(numeric_cols) == 1:
        hist_col = numeric_cols[0]
        fig = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col}",
                           color_discrete_sequence=["#6366f1"], nbins=30)
        fig.update_layout(**CHART_THEME)
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        charts["line"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return charts


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected.", "error")
            return redirect(url_for("index"))
        file = request.files["file"]
        if file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("index"))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            session["filepath"] = filepath
            session["filename"] = filename
            return redirect(url_for("dashboard"))
        flash("Only CSV files are allowed.", "error")
        return redirect(url_for("index"))
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    df = load_dataframe()
    if df is None:
        flash("No dataset loaded. Please upload a CSV file.", "error")
        return redirect(url_for("index"))

    filename = session.get("filename", "dataset.csv")
    rows, cols = df.shape

    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    table_html = df.head(100).to_html(
        classes="data-table", index=False, border=0, na_rep="--"
    )

    charts      = build_charts(df, numeric_cols, categorical_cols)
    insights    = generate_insights(df, numeric_cols, categorical_cols)
    ai_insights = generate_ai_insights(df, numeric_cols, categorical_cols)

    stats = {}
    if numeric_cols:
        stats = df[numeric_cols].describe().round(2).to_dict()

    null_counts = df.isnull().sum()
    null_info   = {col: int(n) for col, n in null_counts.items() if n > 0}

    return render_template(
        "dashboard.html",
        filename=filename,
        rows=rows,
        cols=cols,
        columns=df.columns.tolist(),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        table_html=table_html,
        charts=charts,
        insights=insights,
        ai_insights=ai_insights,
        stats=stats,
        null_info=null_info,
    )


@app.route("/clear")
def clear():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
