"""
app.py — DataLens Flask application.

Responsibilities:
  - Flask setup and configuration
  - File upload handling (route: POST /)
  - Dashboard rendering (route: GET /dashboard)
  - Session reset (route: GET /clear)
  - Natural-language chat API (route: POST /chat)

All analysis logic lives in the `analytics` package:
  analytics.build_charts       → Plotly chart JSON
  analytics.generate_insights  → column stats and auto-notes
  analytics.generate_ai_insights → rule-based AI cards
  analytics.answer_question    → NL query engine
"""

import os

import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

from analytics import (
    answer_question,
    build_charts,
    generate_ai_insights,
    generate_insights,
)

# ── Application setup ──────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Upload directory is created alongside app.py at startup
_UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"]      = _UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB limit

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"csv"})


# ── Internal helpers ───────────────────────────────────────────────────────────

def _is_allowed_file(filename: str) -> bool:
    """Return True if `filename` has a permitted extension (.csv)."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in _ALLOWED_EXTENSIONS
    )


def _load_dataframe() -> pd.DataFrame | None:
    """
    Load the DataFrame stored in the current session.

    Returns None if no file has been uploaded, the session has been
    cleared, or the file can no longer be read from disk.
    """
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    """
    Upload page.

    GET:  Render the upload form.
    POST: Validate and save the uploaded CSV, then redirect to the dashboard.
    """
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            flash("No file selected.", "error")
            return redirect(url_for("index"))

        if not _is_allowed_file(file.filename):
            flash("Only CSV files are allowed.", "error")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        session["filepath"] = filepath
        session["filename"] = filename
        return redirect(url_for("dashboard"))

    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    """
    Main analytics dashboard.

    Loads the uploaded CSV, runs all analysis passes (charts, column
    insights, AI insights), and renders the full dashboard template.
    Redirects back to the upload page if no dataset is in the session.
    """
    df = _load_dataframe()
    if df is None:
        flash("No dataset loaded. Please upload a CSV file.", "error")
        return redirect(url_for("index"))

    filename = session.get("filename", "dataset.csv")
    rows, cols = df.shape

    # Classify columns once; pass lists to all downstream functions
    numeric_cols     = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Run all analysis passes
    charts      = build_charts(df, numeric_cols, categorical_cols)
    insights    = generate_insights(df, numeric_cols, categorical_cols)
    ai_insights = generate_ai_insights(df, numeric_cols, categorical_cols)

    # Descriptive statistics table (numeric columns only)
    stats = (
        df[numeric_cols].describe().round(2).to_dict()
        if numeric_cols else {}
    )

    # Missing-value counts (only columns that actually have nulls)
    null_info = {
        col: int(n)
        for col, n in df.isnull().sum().items()
        if n > 0
    }

    # HTML table for the data-preview tab
    table_html = df.head(100).to_html(
        classes="data-table",
        index=False,
        border=0,
        na_rep="—",
    )

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
    """Clear the current session and return to the upload page."""
    session.clear()
    return redirect(url_for("index"))


@app.route("/chat", methods=["POST"])
def chat():
    """
    Natural-language query endpoint.

    Accepts:  POST JSON body {"question": "<user question>"}
    Returns:  JSON {"question": ..., "answer": ...}

    The heavy lifting is done by analytics.answer_question.
    """
    df = _load_dataframe()
    if df is None:
        return jsonify({"answer": "No dataset loaded. Please upload a CSV file first."})

    body     = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()

    if not question:
        return jsonify({"answer": "Please type a question."})

    try:
        answer = answer_question(df, question)
    except Exception as exc:
        answer = f"Error processing query: {exc}"

    return jsonify({"question": question, "answer": answer})


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
