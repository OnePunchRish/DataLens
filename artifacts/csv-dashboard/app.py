import os
import json
import pandas as pd
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
        else:
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

    table_html = df.head(100).to_html(
        classes="data-table",
        index=False,
        border=0,
        na_rep="—"
    )

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    charts = {}

    if numeric_cols and categorical_cols:
        bar_x = categorical_cols[0]
        bar_y = numeric_cols[0]
        top_vals = df[bar_x].value_counts().head(15).index
        bar_df = df[df[bar_x].isin(top_vals)]
        bar_fig = px.bar(
            bar_df, x=bar_x, y=bar_y,
            title=f"{bar_y} by {bar_x}",
            color=bar_x,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        bar_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title_font_size=16,
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=60)
        )
        bar_fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", tickfont_size=11)
        bar_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        charts["bar"] = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)

    if categorical_cols:
        pie_col = categorical_cols[0]
        pie_counts = df[pie_col].value_counts().head(10)
        pie_fig = px.pie(
            values=pie_counts.values,
            names=pie_counts.index,
            title=f"Distribution of {pie_col}",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        pie_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title_font_size=16,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        pie_fig.update_traces(textfont_color="#1a1a2e")
        charts["pie"] = json.dumps(pie_fig, cls=plotly.utils.PlotlyJSONEncoder)

    if len(numeric_cols) >= 2:
        line_x = numeric_cols[0]
        line_y = numeric_cols[1]
        line_df = df[[line_x, line_y]].dropna().sort_values(line_x).head(200)
        line_fig = px.line(
            line_df, x=line_x, y=line_y,
            title=f"{line_y} over {line_x}",
            color_discrete_sequence=["#6366f1"]
        )
        line_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title_font_size=16,
            margin=dict(l=40, r=20, t=50, b=60)
        )
        line_fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        line_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        charts["line"] = json.dumps(line_fig, cls=plotly.utils.PlotlyJSONEncoder)
    elif len(numeric_cols) == 1:
        hist_col = numeric_cols[0]
        hist_fig = px.histogram(
            df, x=hist_col,
            title=f"Distribution of {hist_col}",
            color_discrete_sequence=["#6366f1"],
            nbins=30
        )
        hist_fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            title_font_size=16,
            margin=dict(l=40, r=20, t=50, b=60)
        )
        hist_fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)")
        hist_fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
        charts["line"] = json.dumps(hist_fig, cls=plotly.utils.PlotlyJSONEncoder)

    stats = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2)
        stats = desc.to_dict()

    null_counts = df.isnull().sum()
    null_info = {col: int(n) for col, n in null_counts.items() if n > 0}

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
        stats=stats,
        null_info=null_info
    )


@app.route("/clear")
def clear():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
