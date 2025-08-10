import os
import io
import re
import html
import unicodedata
import datetime as dt
from typing import Optional, Tuple

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    send_file,
    render_template,
    flash,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
import pandas as pd

# ----------------------------
# Flask + DB setup
# ----------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
    "SQLALCHEMY_DATABASE_URI", "sqlite:///literature.db"  # local default
)
db = SQLAlchemy(app)


# ----------------------------
# Models
# ----------------------------
class Dataset(db.Model):
    __tablename__ = "datasets"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    # ORM cascade so deleting a dataset removes its articles
    articles = db.relationship(
        "Article", backref="dataset", lazy=True, cascade="all, delete-orphan"
    )


class Article(db.Model):
    __tablename__ = "articles"
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(
        db.Integer, db.ForeignKey("datasets.id"), index=True, nullable=False
    )
    title = db.Column(db.Text, nullable=True)
    year = db.Column(db.String(32), nullable=True)
    abstract = db.Column(db.Text, nullable=True)
    extra_json = db.Column(db.JSON, nullable=True)  # remaining columns
    label = db.Column(db.String(8), nullable=True)  # 'yes'|'no'|None
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)


# ----------------------------
# Category model (many-to-many with Article)
# ----------------------------
article_categories = db.Table(
    "article_categories",
    db.Column("article_id", db.Integer, db.ForeignKey("articles.id"), primary_key=True),
    db.Column(
        "category_id", db.Integer, db.ForeignKey("categories.id"), primary_key=True
    ),
)


class Category(db.Model):
    __tablename__ = "categories"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)


# Attach relationship onto Article
Article.categories = db.relationship(
    "Category",
    secondary=article_categories,
    lazy="subquery",
    backref=db.backref("articles", lazy=True),
)

# ----------------------------
# Column detection
# ----------------------------
COLUMN_CANDIDATES = {
    "title": ["title", "paper title", "document_title", "article title", "ti"],
    "year": ["year", "publication year", "pubyear", "date", "yr"],
    "abstract": ["abstract", "summary", "ab", "description"],
}


def normalize_cols(
    df: pd.DataFrame,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
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
    total = (
        db.session.query(func.count(Article.id))
        .filter_by(dataset_id=dataset_id)
        .scalar()
        or 0
    )
    yes = (
        db.session.query(func.count(Article.id))
        .filter_by(dataset_id=dataset_id, label="yes")
        .scalar()
        or 0
    )
    no = (
        db.session.query(func.count(Article.id))
        .filter_by(dataset_id=dataset_id, label="no")
        .scalar()
        or 0
    )
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
        if "doi" in str(k).lower():
            s = str(v or "").strip()
            if not s:
                continue
            s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
            s = s.replace("DOI:", "").replace("doi:", "").strip()
            return s or None
    return None


def extract_source_title(extras: dict | None) -> Optional[str]:
    return _get_by_keys(
        extras,
        keys_like=[
            "source title",
            "journal",
            "conference",
            "booktitle",
            "publication title",
            "venue",
            "source",
        ],
    )


def extract_cited_by(extras: dict | None) -> Optional[int]:
    raw = _get_by_keys(
        extras, keys_like=["cited by", "citations", "times cited", "times-cited"]
    )
    if not raw:
        return None
    m = re.search(r"\d+", raw.replace(",", ""))
    return int(m.group(0)) if m else None


# ----------------------------
# Keyword highlighting (per dataset, stored in session)
# ----------------------------
def _normalize(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "").strip()


def _parse_keywords(q: str) -> list[str]:
    if not q:
        return []
    raw = [t.strip() for t in re.split(r"[,\n;]+", q) if t.strip()]
    seen, out = set(), []
    for t in raw:
        nt = _normalize(t)
        if nt and nt.lower() not in seen:
            seen.add(nt.lower())
            out.append(nt)
    return out


def _terms_for_dataset(dsid: int | None) -> list[str]:
    if not dsid:
        return []
    dct = session.get("highlight_terms", {})
    return dct.get(str(dsid), [])


def _store_terms_for_dataset(dsid: int | None, terms: list[str]) -> None:
    if not dsid:
        return
    dct = session.get("highlight_terms", {})
    dct[str(dsid)] = terms
    session["highlight_terms"] = dct


def _regex_flag_for_dataset(dsid: int | None) -> bool:
    if not dsid:
        return False
    dct = session.get("highlight_regex_flags", {})
    return bool(dct.get(str(dsid), False))


def _store_regex_flag_for_dataset(dsid: int | None, is_regex: bool) -> None:
    if not dsid:
        return
    dct = session.get("highlight_regex_flags", {})
    dct[str(dsid)] = bool(is_regex)
    session["highlight_regex_flags"] = dct


@app.route("/highlight/remove", methods=["POST"])
def remove_highlight():
    dsid = get_active_dataset_id()
    term = (request.form.get("term") or "").strip()
    terms = _terms_for_dataset(dsid)
    if term:
        terms = [t for t in terms if t.lower() != term.lower()]
        _store_terms_for_dataset(dsid, terms)
        flash(f"Removed keyword: {term}", "secondary")
    return redirect(
        request.referrer
        or (url_for("label_next", dataset_id=dsid) if dsid else url_for("upload"))
    )


@app.route("/highlight/clear", methods=["POST"])
def clear_highlight():
    dsid = get_active_dataset_id()
    _store_terms_for_dataset(dsid, [])
    flash("Highlight keywords cleared.", "secondary")
    return redirect(
        request.referrer
        or (url_for("label_next", dataset_id=dsid) if dsid else url_for("upload"))
    )


@app.template_filter("hilite")
def jinja_hilite(text: str, terms: list[str] | None = None, *args, **kwargs) -> str:
    """Highlight only the token(s) that match user patterns.
    Accepts extra args from Jinja without error.
    """
    if not text:
        return ""
    safe = html.escape(text)
    terms = [t for t in (terms or []) if t and t.strip()]
    if not terms:
        return safe

    def looks_like_regex(p: str) -> bool:
        return re.search(r"[.\^\$\+\{\}\(\)\[\]\|\\]", p) is not None

    def build_pattern(p: str):
        p = p.strip()
        if re.fullmatch(r"[*?]+", p):
            return None
        leading_wild = p.startswith("*")
        trailing_wild = p.endswith("*")
        if looks_like_regex(p):
            core = p
        else:
            core = re.escape(p).replace(r"\*", r"[^\s<>]*").replace(r"\?", r"[^\s<>]")
        if not leading_wild:
            core = r"(?<![\w-])" + core
        if not trailing_wild:
            core = core + r"(?![\w-])"
        try:
            return re.compile(core, re.IGNORECASE | re.MULTILINE)
        except re.error:
            return None

    patterns = [rx for t in terms if (rx := build_pattern(t))]
    if not patterns:
        return safe

    segments = re.split(r"(<mark\b[^>]*>|</mark>)", safe, flags=re.IGNORECASE)
    out, in_mark = [], False
    for seg in segments:
        if seg is None or seg == "":
            continue
        if re.fullmatch(r"<mark\b[^>]*>", seg, flags=re.IGNORECASE):
            in_mark = True
            out.append(seg)
            continue
        if re.fullmatch(r"</mark>", seg, flags=re.IGNORECASE):
            in_mark = False
            out.append(seg)
            continue
        if in_mark:
            out.append(seg)
        else:
            text_seg = seg
            for rx in patterns:
                text_seg = rx.sub(
                    lambda m: f'<mark class="kw">{m.group(0)}</mark>', text_seg
                )
            out.append(text_seg)

    return "".join(out)


# ----------------------------
# Navbar context (datasets, counts, keywords, progress)
# ----------------------------
@app.context_processor
def inject_nav():
    dsid = get_active_dataset_id()
    datasets = Dataset.query.order_by(Dataset.created_at.desc()).all()
    if dsid:
        total, yes, no, unlabeled = counts(dsid)
    else:
        total = yes = no = unlabeled = 0
    hl_terms = _terms_for_dataset(dsid)
    hl_is_regex = _regex_flag_for_dataset(dsid)
    return {
        "active_dataset_id": dsid,
        "datasets_for_nav": datasets,
        "nav_counts": {"total": total, "yes": yes, "no": no, "unlabeled": unlabeled},
        "highlight_terms": hl_terms,
        "highlight_is_regex": hl_is_regex,
        "highlight_query": ", ".join(hl_terms),
    }


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
            flash(
                "Could not detect columns. Provide at least 'title' or 'abstract' in your CSV.",
                "danger",
            )
            return redirect(request.url)

        ds = Dataset(
            name=request.form.get("name")
            or f"Dataset {dt.datetime.utcnow():%Y-%m-%d %H:%M}"
        )
        db.session.add(ds)
        db.session.flush()

        for _, row in df.iterrows():
            title = (
                str(row[title_col]).strip()
                if title_col and pd.notna(row[title_col])
                else None
            )
            year = (
                str(row[year_col]).strip()
                if year_col and pd.notna(row[year_col])
                else None
            )
            abstract = (
                str(row[abs_col]).strip()
                if abs_col and pd.notna(row[abs_col])
                else None
            )
            extras = {}
            for c in df.columns:
                if c not in {title_col, year_col, abs_col}:
                    val = row[c]
                    if pd.notna(val):
                        extras[c] = str(val)
            db.session.add(
                Article(
                    dataset_id=ds.id,
                    title=title,
                    year=year,
                    abstract=abstract,
                    extra_json=extras,
                )
            )
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


@app.route("/dataset/<int:dataset_id>/delete", methods=["POST"])
def delete_dataset(dataset_id):
    """
    Delete a dataset and all its articles (uses ORM cascade; includes explicit child delete for safety).
    """
    ds = Dataset.query.get_or_404(dataset_id)

    # Clear active dataset if we're deleting it
    if session.get("dataset_id") == dataset_id:
        session.pop("dataset_id", None)

    # Explicit child delete (paranoid safety for SQLite configs without FK enforcement)
    Article.query.filter_by(dataset_id=dataset_id).delete(synchronize_session=False)
    db.session.delete(ds)
    db.session.commit()

    flash(f"Dataset {dataset_id} deleted.", "success")
    return redirect(url_for("upload"))


@app.route("/highlight", methods=["POST"])
def set_highlight():
    dsid = get_active_dataset_id()
    query = request.form.get("keywords", "")
    is_regex = request.form.get("is_regex") == "on"

    terms = _parse_keywords(query)
    _store_terms_for_dataset(dsid, terms)
    _store_regex_flag_for_dataset(dsid, is_regex)

    # Try compiling to warn about invalid patterns
    bad = []
    if is_regex:
        for t in terms:
            try:
                re.compile(t, re.IGNORECASE | re.MULTILINE)
            except re.error:
                bad.append(t)

    if bad:
        flash(
            f"Updated keywords, but skipped invalid regex: {', '.join(bad)}", "warning"
        )
    else:
        flash("Highlight keywords updated.", "success")

    next_url = request.referrer or (
        url_for("label_next", dataset_id=dsid) if dsid else url_for("upload")
    )
    return redirect(next_url)


@app.route("/label/<int:dataset_id>")
def label_next(dataset_id):
    # If we arrive here with ?skip_id=XYZ, jump to the next unlabeled with a higher id
    skip_id = request.args.get("skip_id", type=int)

    base_q = Article.query.filter_by(dataset_id=dataset_id, label=None).order_by(
        Article.id.asc()
    )

    if skip_id:
        a = base_q.filter(Article.id > skip_id).first()
        if not a:
            # wrap around to the first unlabeled if we were at the end
            a = base_q.first()
    else:
        a = base_q.first()

    total, yes, no, unlabeled = counts(dataset_id)

    doi = extract_doi(a.extra_json) if a else None
    source_title = extract_source_title(a.extra_json) if a else None
    cited_by = extract_cited_by(a.extra_json) if a else None

    pct_done = int(round(((yes + no) / total) * 100)) if total else 0

    return render_template(
        "label.html",
        article=a,
        dataset_id=dataset_id,
        total=total,
        yes=yes,
        no=no,
        unlabeled=unlabeled,
        doi=doi,
        source_title=source_title,
        cited_by=cited_by,
        pct_done=pct_done,
        categories=Category.query.order_by(
            Category.name.asc()
        ).all(),  # for category selector
    )


# review route
@app.route("/review/<int:dataset_id>")
def review(dataset_id):
    status = request.args.get("status", "all")  # all|yes|no|unlabeled
    hide = request.args.get("hide")  # 'labeled' or None

    q = Article.query.filter_by(dataset_id=dataset_id)

    if status == "yes":
        q = q.filter_by(label="yes")
    elif status == "no":
        q = q.filter_by(label="no")
    elif status == "unlabeled":
        q = q.filter(Article.label.is_(None))
    else:
        # status == 'all'
        if hide == "labeled":
            q = q.filter(Article.label.is_(None))  # hide labeled in 'all'

    rows = q.order_by(Article.id.asc()).all()
    total, yes, no, unlabeled = counts(dataset_id)
    return render_template(
        "review.html",
        rows=rows,
        dataset_id=dataset_id,
        status=status,
        total=total,
        yes=yes,
        no=no,
        unlabeled=unlabeled,
        hide=hide,
    )


@app.route("/relabel/<int:article_id>", methods=["POST"])
def relabel(article_id):
    a = Article.query.get_or_404(article_id)

    new_label = request.form.get("label")  # yes|no|clear
    if new_label == "clear":
        a.label = None
    elif new_label in ("yes", "no"):
        a.label = new_label

    notes = request.form.get("notes")
    if notes is not None:
        a.notes = notes

    db.session.commit()

    # preserve current filters so the row can disappear if hidden
    status = request.args.get("status", "all")
    hide = request.args.get("hide")

    return redirect(
        url_for("review", dataset_id=a.dataset_id, status=status, hide=hide)
    )


@app.route("/label/submit/<int:article_id>", methods=["POST"])
def label_submit(article_id):
    a = Article.query.get_or_404(article_id)
    decision = request.form.get("decision")  # "yes" | "no" | "skip"
    notes = request.form.get("notes")

    if decision in ("yes", "no"):
        a.label = decision
        if notes is not None:
            a.notes = notes
        db.session.commit()
        return redirect(url_for("label_next", dataset_id=a.dataset_id))

    if decision == "skip":
        # don’t change label; jump to the next unlabeled with higher id
        return redirect(url_for("label_next", dataset_id=a.dataset_id, skip_id=a.id))

    # default safety: reload current article
    return redirect(url_for("label_next", dataset_id=a.dataset_id))


# ----------------------------
# Category routes
# ----------------------------
@app.post("/articles/<int:article_id>/categorize")
def categorize(article_id):
    """
    Assign an article to an existing category, or create a new category and assign it.
    Expects:
      - action = "assign" or "create_assign"
      - category_id (when action == "assign")
      - new_category (when action == "create_assign")
    """
    a = Article.query.get_or_404(article_id)
    action = request.form.get("action")
    cat_id = request.form.get("category_id")
    new_name = (request.form.get("new_category") or "").strip()

    if action == "assign" and cat_id:
        c = Category.query.get_or_404(int(cat_id))
        if c not in a.categories:
            a.categories.append(c)
            db.session.commit()
            flash(f"Added to category: {c.name}", "success")
        else:
            flash(f"Already in category: {c.name}", "secondary")

    elif action == "create_assign" and new_name:
        # fetch existing by name (case-insensitive) or create
        c = Category.query.filter(func.lower(Category.name) == new_name.lower()).first()
        if not c:
            c = Category(name=new_name)
            db.session.add(c)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                c = Category.query.filter(
                    func.lower(Category.name) == new_name.lower()
                ).first()
        if c not in a.categories:
            a.categories.append(c)
            db.session.commit()
            flash(f"Created and assigned: {c.name}", "success")
        else:
            flash(f"Already in category: {c.name}", "secondary")

    else:
        flash("No category action performed.", "warning")

    return redirect(request.referrer or url_for("label_next", dataset_id=a.dataset_id))


@app.post("/articles/<int:article_id>/categories/<int:category_id>/remove")
def remove_category(article_id, category_id):
    a = Article.query.get_or_404(article_id)
    c = Category.query.get_or_404(category_id)
    if c in a.categories:
        a.categories.remove(c)
        db.session.commit()
        flash(f"Removed category: {c.name}", "info")
    return redirect(request.referrer or url_for("label_next", dataset_id=a.dataset_id))


# ----------------------------
# Export
# ----------------------------
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
        # categories as a semicolon list
        if a.categories:
            row["categories"] = "; ".join(sorted([c.name for c in a.categories]))
        else:
            row["categories"] = ""
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


@app.route("/categories/<int:dataset_id>")
def categories(dataset_id):
    # Todas as categorias existentes (ordenadas)
    cats = Category.query.order_by(Category.name.asc()).all()

    # IDs de categorias selecionadas via query string: ?cat=1&cat=3
    selected = request.args.getlist("cat", type=int)

    # Artigos do dataset
    q = Article.query.filter_by(dataset_id=dataset_id)

    # Se houver seleção, filtra artigos que pertencem a TODAS as categorias selecionadas
    # (interseção). Para um filtro por "qualquer das selecionadas", substitua por .filter(Category.id.in_(selected)).
    if selected:
        for cid in selected:
            q = q.join(Article.categories).filter(Category.id == cid)
        q = q.distinct()

    rows = q.order_by(Article.id.asc()).all()

    # Contagem por categoria restrita ao dataset atual
    counts_by_cat = (
        db.session.query(Category.id, func.count(Article.id))
        .select_from(Category)
        .join(Category.articles)
        .filter(Article.dataset_id == dataset_id)
        .group_by(Category.id)
        .all()
    )
    counts_map = {cid: cnt for cid, cnt in counts_by_cat}

    total, yes, no, unlabeled = counts(dataset_id)

    return render_template(
        "categories.html",
        dataset_id=dataset_id,
        categories=cats,
        selected=selected,
        rows=rows,
        counts_map=counts_map,
        total=total,
        yes=yes,
        no=no,
        unlabeled=unlabeled,
    )


# ----------------------------
# CLI entry
# ----------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
