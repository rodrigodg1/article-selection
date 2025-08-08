import os
import io
import re
import json
import datetime as dt
from typing import Optional, Tuple, List, Dict

from flask import (
    Flask, request, redirect, url_for, send_file,
    render_template, flash, session
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
import pandas as pd

# ----------------------------
# Flask + DB setup
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///literature.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# ----------------------------
# Models
# ----------------------------
class Dataset(db.Model):
    __tablename__ = "datasets"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

class Article(db.Model):
    __tablename__ = "articles"
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey("datasets.id"), index=True, nullable=False)
    title = db.Column(db.Text, nullable=True)
    year = db.Column(db.String(32), nullable=True)
    abstract = db.Column(db.Text, nullable=True)
    extra_json = db.Column(db.JSON, nullable=True)  # remaining columns + PMR cache
    label = db.Column(db.String(8), nullable=True)  # 'yes'|'no'|None
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    dataset = db.relationship("Dataset", backref=db.backref("articles", lazy=True))

# ----------------------------
# Column detection
# ----------------------------
COLUMN_CANDIDATES = {
    "title": ["title", "paper title", "document_title", "article title", "ti"],
    "year": ["year", "publication year", "pubyear", "date", "yr"],
    "abstract": ["abstract", "summary", "ab", "description"],
}

def normalize_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    cols = {c.lower().strip(): c for c in df.columns}
    def find_one(kind):
        for opt in COLUMN_CANDIDATES[kind]:
            if opt in cols:
                return cols[opt]
        return None
    return find_one("title"), find_one("year"), find_one("abstract")

def get_active_dataset_id() -> Optional[int]:
    dsid = session.get("dataset_id")
    if dsid is None:
        ds = Dataset.query.order_by(Dataset.id.desc()).first()
        return ds.id if ds else None
    return dsid

def counts(dataset_id: int):
    total = db.session.query(func.count(Article.id)).filter_by(dataset_id=dataset_id).scalar() or 0
    yes = db.session.query(func.count(Article.id)).filter_by(dataset_id=dataset_id, label="yes").scalar() or 0
    no = db.session.query(func.count(Article.id)).filter_by(dataset_id=dataset_id, label="no").scalar() or 0
    unlabeled = total - yes - no
    return total, yes, no, unlabeled

# ----------------------------
# DOI / Source / Cited-by extractors
# ----------------------------
def _get_by_keys(extras: dict | None, keys_like: list[str]) -> Optional[str]:
    if not extras:
        return None
    for k, v in extras.items():
        lk = str(k).lower()
        if any(s in lk for s in keys_like):
            s = str(v or "").strip()
            return s or None
    return None

def extract_doi(extras: Optional[dict]) -> Optional[str]:
    if not extras:
        return None
    for k, v in extras.items():
        if 'doi' in str(k).lower():
            s = str(v or '').strip()
            if not s:
                continue
            s = s.replace('https://doi.org/', '').replace('http://doi.org/', '')
            s = s.replace('DOI:', '').replace('doi:', '').strip()
            return s or None
    return None

def extract_source_title(extras: dict | None) -> Optional[str]:
    return _get_by_keys(extras, keys_like=[
        "source title", "journal", "conference", "booktitle", "publication title", "venue", "source"
    ])

def extract_cited_by(extras: dict | None) -> Optional[int]:
    raw = _get_by_keys(extras, keys_like=["cited by", "citations", "times cited", "times-cited"])
    if not raw:
        return None
    m = re.search(r"\d+", raw.replace(",", ""))
    return int(m.group(0)) if m else None

# ----------------------------
# Improved heuristic PMR (no LLM)
# ----------------------------

# 1) Structured abstracts: headings to buckets
SECTION_ALIASES: Dict[str, List[str]] = {
    "problem": ["background", "introduction", "objective", "objectives", "aim", "aims",
                "purpose", "motivation", "problem", "gap"],
    "method":  ["methods", "materials and methods", "approach", "methodology", "design",
                "proposed method", "proposed approach", "framework"],
    "results": ["results", "findings", "evaluation", "experiments", "experiment",
                "conclusion", "conclusions", "outcomes", "performance"]
}

HEADING_RE = re.compile(
    r"(?mi)^\s*(Background|Introduction|Objectives?|Aims?|Purpose|Motivation|Problem|Gap|"
    r"Methods?|Materials and Methods|Approach|Methodology|Design|Proposed (?:Method|Approach)|Framework|"
    r"Results?|Findings?|Evaluation|Experiments?|Conclusions?|Outcomes?|Performance)\s*:\s*"
)

# 2) Weighted cue sets for sentence scoring (lowercase substrings → weights)
PROBLEM_WEIGHTS: Dict[str, int] = {
    "problem":3, "challenge":3, "gap":3, "issue":2, "limitation":2, "bottleneck":2,
    "lack":2, "scarcity":2, "need":2, "motivation":2, "objective":2, "aim":2, "goal":2,
    "research question":3, "we address":3, "we tackle":3, "we investigate":2,
    "this paper addresses":3, "this work addresses":3
}
METHOD_WEIGHTS: Dict[str, int] = {
    "we propose":4, "we present":3, "we introduce":3, "we develop":3, "we design":3,
    "this paper proposes":4, "this work proposes":4,
    "method":2, "approach":2, "framework":2, "algorithm":3, "model":2, "architecture":2,
    "pipeline":2, "procedure":2, "protocol":3, "scheme":2, "system":2, "implementation":2,
    "dataset":1, "training":1, "inference":1, "evaluation protocol":2
}
RESULTS_WEIGHTS: Dict[str, int] = {
    "we show":3, "we demonstrate":3, "we find":2, "we observe":2, "we report":2,
    "results indicate":3, "results show":3, "experiments show":3, "evaluation shows":3,
    "achieve":2, "achieves":2, "achieved":2, "outperform":3, "improve":2, "improves":2, "improved":2,
    "state-of-the-art":3, "sota":3, "significant":2, "statistically significant":3,
    "accuracy":1, "precision":1, "recall":1, "f1":1, "f1-score":1, "auroc":2, "auc":1, "rmse":1,
    "%":1, "percent":1, "p<":2, "p <":2
}

# Sentence splitting with simple abbreviation handling
ABBREV_FIX = {
    "e.g.": "eg", "i.e.": "ie", "et al.": "et al", "Fig.": "Fig", "fig.": "fig",
    "Dr.": "Dr", "Mr.": "Mr", "Ms.": "Ms", "Prof.": "Prof", "vs.": "vs",
}
_SENT_SPLIT = re.compile(r"(?<!\b[A-Z][a-z])(?<=[.!?])\s+(?=[A-Z(])")

def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def _pretokenize(text: str) -> str:
    t = text
    for k, v in ABBREV_FIX.items():
        t = t.replace(k, v)
    return t

def _split_sentences(text: str) -> List[str]:
    t = _pretokenize(text)
    parts = _SENT_SPLIT.split(_normalize_spaces(t))
    return [p.strip() for p in parts if p.strip()]

def _soft_clip(s: str, max_len: int = 280) -> str:
    s = s.strip()
    if len(s) <= max_len:
        return s
    cut = s.rfind(". ", 0, max_len)
    if cut == -1:
        cut = s.rfind("; ", 0, max_len)
    if cut == -1:
        cut = s.rfind(", ", 0, max_len)
    if cut == -1:
        cut = s.rfind(" ", 0, max_len)
    if cut == -1:
        cut = max_len
    return s[:cut].rstrip(" ,;") + "…"

def _score_sentence(s: str, weights: Dict[str, int]) -> int:
    ls = s.lower()
    score = 0
    for term, w in weights.items():
        if term in ls:
            score += w
    # Bonus if sentence length suggests content
    tok_count = len(ls.split())
    if 10 <= tok_count <= 50:
        score += 1
    # Extra bonus for numbers in results
    if weights is RESULTS_WEIGHTS and re.search(r"\d|%", ls):
        score += 1
    return score

def _pick_best_weighted(sentences: List[str], weights: Dict[str, int]) -> Optional[str]:
    if not sentences:
        return None
    best_idx, best_score = -1, -10**9
    for i, s in enumerate(sentences):
        sc = _score_sentence(s, weights)
        if sc > best_score:
            best_idx, best_score = i, sc
    if best_score <= 0:
        return None
    return sentences[best_idx].strip()

def _parse_structured_sections(text: str) -> Optional[Dict[str, str]]:
    """
    Parse 'Background: ... Methods: ... Results: ...' style abstracts.
    Uses regex finditer to avoid split-capture pitfalls.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return None
    sec_map: Dict[str, List[str]] = {"problem": [], "method": [], "results": []}
    for idx, m in enumerate(matches):
        label = m.group(1).lower()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if not content:
            continue
        # Map heading to bucket
        for bucket, aliases in SECTION_ALIASES.items():
            if any(a in label for a in aliases):
                sec_map[bucket].append(content)
                break
    out = {}
    for k, vs in sec_map.items():
        if vs:
            out[k] = _soft_clip(_normalize_spaces(" ".join(vs)))
    return out if out else None

def summarize_pmr(abstract: Optional[str]) -> dict:
    """
    Heuristic PMR extractor:
    1) Try structured abstract headings.
    2) Otherwise, score sentences for each bucket.
    3) Backoffs (first/middle/last; prefer numeric tails for results).
    """
    if not abstract or not abstract.strip():
        return {"problem": None, "method": None, "results": None}

    text = abstract.strip()

    # 1) Structured abstracts
    structured = _parse_structured_sections(text)
    if structured:
        return {
            "problem": structured.get("problem"),
            "method": structured.get("method"),
            "results": structured.get("results"),
        }

    # 2) Sentence-based scoring
    sentences = _split_sentences(text)
    if not sentences:
        return {"problem": None, "method": None, "results": None}

    problem = _pick_best_weighted(sentences, PROBLEM_WEIGHTS)
    method  = _pick_best_weighted(sentences, METHOD_WEIGHTS)
    results = _pick_best_weighted(sentences, RESULTS_WEIGHTS)

    # 3) Backoffs
    n = len(sentences)
    if not problem:
        problem = sentences[0].strip()
    if not method:
        mid_idx = min(1, n - 1) if n <= 3 else n // 2
        method = sentences[mid_idx].strip()
    if not results:
        tail = sentences[-3:] if n >= 3 else sentences[-1:]
        numeric = next((s for s in tail if re.search(r"\d|%", s)), None)
        results = (numeric or tail[-1]).strip()

    # Clip for readability
    return {
        "problem": _soft_clip(problem),
        "method": _soft_clip(method),
        "results": _soft_clip(results),
    }

def ensure_pmr_cached(article: Article, force: bool = False) -> Tuple[dict, str]:
    """
    Compute PMR via heuristic and cache into extra_json['pmr'] when it has content.
    - force=True recomputes even if cache exists.
    Returns (pmr_dict, 'heuristic' or 'cache').
    """
    extras = article.extra_json or {}
    cached = extras.get("pmr")

    def has_content(d: Optional[dict]) -> bool:
        return isinstance(d, dict) and any(d.get(k) for k in ("problem", "method", "results"))

    if (not force) and has_content(cached):
        return cached, "cache"

    pmr = summarize_pmr(article.abstract or "")
    if has_content(pmr):
        extras["pmr"] = pmr
        article.extra_json = extras
        db.session.commit()
        return pmr, "heuristic"
    else:
        # Don't cache empties; allow future retries
        return pmr, "heuristic"

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    dsid = get_active_dataset_id()
    if dsid:
        return redirect(url_for("label_next", dataset_id=dsid))
    return redirect(url_for("upload"))

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files or request.files["file"].filename == "":
            flash("Please choose a CSV file.", "warning")
            return redirect(request.url)

        f = request.files["file"]
        try:
            df = pd.read_csv(f)
        except Exception as e:
            flash(f"Failed to read CSV: {e}", "danger")
            return redirect(request.url)

        title_col, year_col, abs_col = normalize_cols(df)
        if not (title_col or abs_col):
            flash("Could not detect columns. Provide at least 'title' or 'abstract' in your CSV.", "danger")
            return redirect(request.url)

        ds = Dataset(name=request.form.get("name") or f"Dataset {dt.datetime.utcnow():%Y-%m-%d %H:%M}")
        db.session.add(ds)
        db.session.flush()

        for _, row in df.iterrows():
            title = str(row[title_col]).strip() if title_col and pd.notna(row[title_col]) else None
            year = str(row[year_col]).strip() if year_col and pd.notna(row[year_col]) else None
            abstract = str(row[abs_col]).strip() if abs_col and pd.notna(row[abs_col]) else None
            extras = {}
            for c in df.columns:
                if c not in {title_col, year_col, abs_col}:
                    val = row[c]
                    if pd.notna(val):
                        extras[c] = str(val)
            db.session.add(Article(dataset_id=ds.id, title=title, year=year, abstract=abstract, extra_json=extras))
        db.session.commit()
        session["dataset_id"] = ds.id
        flash("CSV uploaded successfully.", "success")
        return redirect(url_for("label_next", dataset_id=ds.id))

    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    return render_template("upload.html", datasets=datasets)

@app.route("/switch/<int:dataset_id>")
def switch(dataset_id):
    if Dataset.query.get(dataset_id) is None:
        flash("Dataset not found.", "warning")
        return redirect(url_for("upload"))
    session["dataset_id"] = dataset_id
    return redirect(url_for("label_next", dataset_id=dataset_id))

@app.route("/label/<int:dataset_id>")
def label_next(dataset_id):
    a = Article.query.filter_by(dataset_id=dataset_id, label=None).order_by(Article.id.asc()).first()
    total, yes, no, unlabeled = counts(dataset_id)

    doi = extract_doi(a.extra_json) if a else None
    source_title = extract_source_title(a.extra_json) if a else None
    cited_by = extract_cited_by(a.extra_json) if a else None

    pct_done = int(round(((yes + no) / total) * 100)) if total else 0

    pmr = {"problem": None, "method": None, "results": None}
    pmr_source = "none"
    if a:
        force = request.args.get("force") == "1"
        pmr, pmr_source = ensure_pmr_cached(a, force=force)

    return render_template(
        "label.html",
        article=a, dataset_id=dataset_id,
        total=total, yes=yes, no=no, unlabeled=unlabeled,
        doi=doi, source_title=source_title, cited_by=cited_by,
        pct_done=pct_done, pmr=pmr, pmr_source=pmr_source
    )

@app.route("/label/submit/<int:article_id>", methods=["POST"])
def label_submit(article_id):
    a = Article.query.get_or_404(article_id)
    decision = request.form.get("decision")  # yes/no/skip
    if decision in ("yes", "no"):
        a.label = decision
        a.notes = request.form.get("notes") or a.notes
        db.session.commit()
    elif decision == "skip":
        pass
    return redirect(url_for("label_next", dataset_id=a.dataset_id))

@app.route("/review/<int:dataset_id>")
def review(dataset_id):
    status = request.args.get("status", "all")  # all|yes|no|unlabeled
    q = Article.query.filter_by(dataset_id=dataset_id)
    if status == "yes":
        q = q.filter_by(label="yes")
    elif status == "no":
        q = q.filter_by(label="no")
    elif status == "unlabeled":
        q = q.filter_by(label=None)
    rows = q.order_by(Article.id.asc()).all()
    total, yes, no, unlabeled = counts(dataset_id)
    return render_template(
        "review.html", rows=rows, dataset_id=dataset_id, status=status,
        total=total, yes=yes, no=no, unlabeled=unlabeled
    )

@app.route("/relabel/<int:article_id>", methods=["POST"])
def relabel(article_id):
    a = Article.query.get_or_404(article_id)
    new_label = request.form.get("label")  # yes|no|clear
    if new_label == "clear":
        a.label = None
    elif new_label in ("yes", "no"):
        a.label = new_label
    a.notes = request.form.get("notes") or a.notes
    db.session.commit()
    return redirect(url_for("review", dataset_id=a.dataset_id, status=request.args.get("status", "all")))

@app.route("/export/<int:dataset_id>")
def export(dataset_id):
    q = Article.query.filter_by(dataset_id=dataset_id).order_by(Article.id.asc())
    data = []
    for a in q:
        row = {
            "id": a.id,
            "title": a.title,
            "year": a.year,
            "abstract": a.abstract,
            "label": a.label,
            "notes": a.notes,
        }
        if a.extra_json:
            row.update({f"extra::{k}": v for k, v in a.extra_json.items()})
        data.append(row)
    df = pd.DataFrame(data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"dataset_{dataset_id}_labeled.csv",
    )

@app.route("/debug/pmr")
def debug_pmr():
    sample_abs = (
        "Background: Cross-silo hospitals need collaborative modeling without exposing patient data. "
        "Methods: We design a federated learning protocol with secure aggregation and differential privacy. "
        "Results: On real EHRs, our approach improves AUROC by 8–12% over centralized baselines while complying with HIPAA."
    )
    out = summarize_pmr(sample_abs)
    return {"source": "heuristic", "pmr": out}, 200, {"Content-Type": "application/json"}

# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
