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


# ── helpers ──────────────────────────────────────────────────────────────────

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


# ── insights engine ──────────────────────────────────────────────────────────

def detect_trend(series: pd.Series) -> dict:
    """
    Split the (non-null) series in half and compare means.
    Returns direction ('up', 'down', 'flat') and percentage change.
    """
    clean = series.dropna()
    if len(clean) < 10:
        return {"direction": "flat", "change_pct": 0.0}

    mid = len(clean) // 2
    first_mean = clean.iloc[:mid].mean()
    second_mean = clean.iloc[mid:].mean()

    if first_mean == 0:
        return {"direction": "flat", "change_pct": 0.0}

    change_pct = round((second_mean - first_mean) / abs(first_mean) * 100, 1)

    if change_pct > 5:
        direction = "up"
    elif change_pct < -5:
        direction = "down"
    else:
        direction = "flat"

    return {"direction": direction, "change_pct": change_pct}


def numeric_column_insights(df: pd.DataFrame, col: str) -> dict:
    """Compute extended statistics for a single numeric column."""
    s = df[col].dropna()
    if s.empty:
        return {}

    trend = detect_trend(s)

    return {
        "mean":   round(float(s.mean()), 3),
        "median": round(float(s.median()), 3),
        "std":    round(float(s.std()), 3),
        "min":    round(float(s.min()), 3),
        "max":    round(float(s.max()), 3),
        "range":  round(float(s.max() - s.min()), 3),
        "skew":   round(float(s.skew()), 2),
        "trend":  trend,
        "non_null": int(s.count()),
    }


def top_values_for_column(df: pd.DataFrame, col: str, n: int = 5) -> list:
    """Return top-n value counts as a list of dicts with value, count, pct."""
    total = len(df)
    counts = df[col].value_counts(dropna=True).head(n)
    result = []
    for val, cnt in counts.items():
        result.append({
            "value": str(val),
            "count": int(cnt),
            "pct":   round(cnt / total * 100, 1),
        })
    return result


def generate_auto_notes(df: pd.DataFrame, numeric_cols, categorical_cols,
                         numeric_insights, top_values) -> list:
    """Produce a list of plain-English insight sentences about the dataset."""
    notes = []
    rows = len(df)

    # Highest-cardinality categorical column
    if categorical_cols:
        unique_counts = {c: df[c].nunique() for c in categorical_cols}
        most_varied = max(unique_counts, key=unique_counts.get)
        notes.append(
            f'"{most_varied}" has the most variety — '
            f'{unique_counts[most_varied]:,} unique values across {rows:,} rows.'
        )

    # Dominant category in first categorical column
    if top_values:
        first_cat = next(iter(top_values))
        top = top_values[first_cat]
        if top:
            notes.append(
                f'The most common "{first_cat}" is '
                f'"{top[0]["value"]}" ({top[0]["pct"]}% of rows).'
            )

    # Numeric trends
    for col, info in numeric_insights.items():
        trend = info.get("trend", {})
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

    # Skewness alerts
    for col, info in numeric_insights.items():
        skew = info.get("skew", 0)
        if abs(skew) > 1.5:
            direction = "right (positively)" if skew > 0 else "left (negatively)"
            notes.append(
                f'"{col}" is heavily skewed {direction} '
                f'(skewness = {skew}), suggesting outliers or a long tail.'
            )

    # Wide spread
    for col, info in numeric_insights.items():
        mean = info.get("mean", 0)
        std = info.get("std", 0)
        if mean and std and std / abs(mean) > 1:
            notes.append(
                f'"{col}" has high variability — '
                f'the standard deviation ({std}) is larger than the mean ({mean}).'
            )

    return notes[:6]  # cap at 6 notes to stay clean


def generate_insights(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    """
    Main entry point for the insights engine.
    Returns a dict consumed by the Jinja2 template.
    """
    # Pick key columns to feature (up to 4 categorical, all numeric up to 6)
    featured_cat = categorical_cols[:4]
    featured_num = numeric_cols[:6]

    top_values = {col: top_values_for_column(df, col) for col in featured_cat}

    numeric_insights = {col: numeric_column_insights(df, col) for col in featured_num}

    auto_notes = generate_auto_notes(df, numeric_cols, categorical_cols,
                                      numeric_insights, top_values)

    return {
        "top_values":       top_values,       # {col: [{value, count, pct}]}
        "numeric_insights": numeric_insights,  # {col: {mean, median, std, …}}
        "auto_notes":       auto_notes,        # [str]
    }


# ── chart builder ─────────────────────────────────────────────────────────────

CHART_THEME = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    title_font_size=16,
    margin=dict(l=40, r=20, t=50, b=60),
)

AXIS_STYLE = dict(gridcolor="rgba(255,255,255,0.08)")


def build_charts(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> dict:
    charts = {}

    if numeric_cols and categorical_cols:
        bar_x = categorical_cols[0]
        bar_y = numeric_cols[0]
        top_vals = df[bar_x].value_counts().head(15).index
        bar_df = df[df[bar_x].isin(top_vals)]
        fig = px.bar(
            bar_df, x=bar_x, y=bar_y,
            title=f"{bar_y} by {bar_x}",
            color=bar_x,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig.update_layout(**CHART_THEME, showlegend=False)
        fig.update_xaxes(**AXIS_STYLE, tickfont_size=11)
        fig.update_yaxes(**AXIS_STYLE)
        charts["bar"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    if categorical_cols:
        pie_col = categorical_cols[0]
        counts = df[pie_col].value_counts().head(10)
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            title=f"Distribution of {pie_col}",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig.update_layout(**{**CHART_THEME, "margin": dict(l=20, r=20, t=50, b=20)})
        fig.update_traces(textfont_color="#1a1a2e")
        charts["pie"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    if len(numeric_cols) >= 2:
        line_x, line_y = numeric_cols[0], numeric_cols[1]
        line_df = df[[line_x, line_y]].dropna().sort_values(line_x).head(200)
        fig = px.line(
            line_df, x=line_x, y=line_y,
            title=f"{line_y} over {line_x}",
            color_discrete_sequence=["#6366f1"],
        )
        fig.update_layout(**CHART_THEME)
        fig.update_xaxes(**AXIS_STYLE)
        fig.update_yaxes(**AXIS_STYLE)
        charts["line"] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    elif len(numeric_cols) == 1:
        hist_col = numeric_cols[0]
        fig = px.histogram(
            df, x=hist_col,
            title=f"Distribution of {hist_col}",
            color_discrete_sequence=["#6366f1"],
            nbins=30,
        )
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
        classes="data-table", index=False, border=0, na_rep="—"
    )

    charts   = build_charts(df, numeric_cols, categorical_cols)
    insights = generate_insights(df, numeric_cols, categorical_cols)

    # Legacy stats dict (used by existing Statistics tab)
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
