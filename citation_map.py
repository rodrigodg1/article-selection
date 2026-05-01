"""
Build graph data for the citation / similarity map view.

Citation edges: DOIs found in "reference-like" extra columns are matched to
other articles' own DOI (same dataset). Typical for exports that include a
cited-references blob (e.g. Web of Science 'CR').

Similarity edges: Jaccard on title+abstract tokens (fallback / supplement).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple


class _ArticleLike(Protocol):
    id: int
    title: Optional[str]
    year: Optional[str]
    abstract: Optional[str]
    label: Optional[str]
    extra_json: Optional[Dict[str, Any]]


DOI_PATTERN = re.compile(r"10\.\d{4,9}/[^\s;,\]\"'<>]+", re.IGNORECASE)

# Extra column names (substring match, lowercased) treated as reference lists
REF_KEY_HINTS = (
    "reference",
    "cited ref",
    "bibliography",
    "cited reference",
    "cited works",
    "cr",
)


def _normalize_doi(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = str(s).strip()
    s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
    s = s.replace("doi:", "").replace("DOI:", "").strip().rstrip(").,;]")
    return s.lower() or None


def _extract_doi_from_extras(extras: Optional[Dict[str, Any]]) -> Optional[str]:
    if not extras:
        return None
    for k, v in extras.items():
        lk = str(k).lower().strip()
        # WoS often uses column name "DI" for DOI
        if "doi" in lk or lk == "di":
            s = str(v or "").strip()
            if not s:
                continue
            s = s.replace("https://doi.org/", "").replace("http://doi.org/", "")
            s = s.replace("DOI:", "").replace("doi:", "").strip()
            return s or None
    return None


def _doi_from_article(a: _ArticleLike) -> Optional[str]:
    d = _extract_doi_from_extras(a.extra_json)
    if d:
        return _normalize_doi(d)
    return None


def _citation_empty_hint(rows: List[_ArticleLike], n_citation_edges: int) -> Optional[str]:
    """Short PT explanation when no within-CSV citation arrows are possible."""
    if n_citation_edges > 0 or len(rows) < 2:
        return None
    n_doi = sum(1 for a in rows if _doi_from_article(a))
    n_ref = sum(1 for a in rows if (_reference_blob(a.extra_json) or "").strip())
    parts: List[str] = []
    if n_doi < 2:
        parts.append(
            "Menos de dois artigos têm DOI reconhecido nas colunas extra "
            "(o nome da coluna deve conter «doi» ou ser «DI», como na Web of Science)."
        )
    if n_ref == 0:
        parts.append(
            "Não há coluna de referências reconhecida (ex.: CR, References, Cited References, Bibliography)."
        )
    if n_doi >= 2 and n_ref > 0:
        parts.append(
            "Com DOI e referências, as setas azuis só aparecem quando o texto de referências "
            "inclui o DOI de outro registo deste mesmo CSV — muitas vezes as citações são só para trabalhos fora da exportação."
        )
    return " ".join(parts) if parts else None


def _reference_blob(extras: Optional[Dict[str, Any]]) -> str:
    if not extras:
        return ""
    parts: List[str] = []
    for k, v in extras.items():
        lk = str(k).lower().strip()
        if lk == "cr" or any(h in lk for h in REF_KEY_HINTS):
            if v is not None and str(v).strip():
                parts.append(str(v))
    return "\n".join(parts)


def _dois_in_text(text: str) -> Set[str]:
    out: Set[str] = set()
    for m in DOI_PATTERN.finditer(text or ""):
        nd = _normalize_doi(m.group(0))
        if nd:
            out.add(nd)
    return out


def _token_set(title: Optional[str], abstract: Optional[str]) -> Set[str]:
    blob = f"{title or ''} {abstract or ''}".lower()
    return {w for w in re.findall(r"[a-z0-9]+", blob) if len(w) >= 3}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _screening_label_pt(label: Optional[str]) -> str:
    if label == "yes":
        return "Incluir (sim)"
    if label == "no":
        return "Excluir (não)"
    return "Sem rótulo"


def _category_names(a: Any) -> List[str]:
    cats = getattr(a, "categories", None)
    if not cats:
        return []
    try:
        return sorted({str(getattr(c, "name", c)) for c in cats})
    except TypeError:
        return []


MAX_SIM_ARTICLES = 220  # full pairwise Jaccard cap
TOP_K_SIM = 4
DEFAULT_MIN_SIM = 0.06


def build_map_payload(
    rows: List[_ArticleLike],
    min_similarity: float = DEFAULT_MIN_SIM,
    top_k_similar: int = TOP_K_SIM,
    include_similarity: bool = True,
) -> Dict[str, Any]:

    if not rows:
        return {
            "nodes": [],
            "edges": [],
            "stats": {
                "n_articles": 0,
                "n_citation_edges": 0,
                "n_similarity_edges": 0,
                "similarity_skipped": False,
                "message": "Não há artigos neste conjunto.",
            },
        }

    doi_to_id: Dict[str, int] = {}
    for a in rows:
        d = _doi_from_article(a)
        if d and d not in doi_to_id:
            doi_to_id[d] = a.id

    citation_pairs: Set[Tuple[int, int]] = set()
    citation_edges: List[Dict[str, Any]] = []

    for a in rows:
        blob = _reference_blob(a.extra_json)
        if not blob:
            continue
        for target_doi in _dois_in_text(blob):
            tid = doi_to_id.get(target_doi)
            if tid is None or tid == a.id:
                continue
            key = (a.id, tid)
            if key in citation_pairs:
                continue
            citation_pairs.add(key)
            citation_edges.append(
                {
                    "from": a.id,
                    "to": tid,
                    "type": "citation",
                    "value": 1.0,
                    "title": (
                        "Citação (neste CSV): o registo da seta referencia, "
                        "nas colunas de bibliografia/CR, o DOI do registo apontado."
                    ),
                }
            )

    sim_edges: List[Dict[str, Any]] = []
    similarity_skipped = False

    if include_similarity and len(rows) <= MAX_SIM_ARTICLES:
        tokens = {a.id: _token_set(a.title, a.abstract) for a in rows}
        used_pairs: Set[Tuple[int, int]] = set()
        # undirected: store (min,max)
        for a in rows:
            scores: List[Tuple[float, int]] = []
            ta = tokens[a.id]
            for b in rows:
                if b.id == a.id:
                    continue
                j = _jaccard(ta, tokens[b.id])
                if j >= min_similarity:
                    scores.append((j, b.id))
            scores.sort(reverse=True)
            for j, bid in scores[:top_k_similar]:
                p = (min(a.id, bid), max(a.id, bid))
                if p in used_pairs:
                    continue
                # skip if we already have a citation edge in either direction
                if (a.id, bid) in citation_pairs or (bid, a.id) in citation_pairs:
                    continue
                used_pairs.add(p)
                sim_edges.append(
                    {
                        "from": a.id,
                        "to": bid,
                        "type": "similarity",
                        "value": round(j, 4),
                        "title": (
                            f"Semelhança de texto: Jaccard ≈ {100 * j:.1f}% "
                            f"entre título+resumo (top‑{top_k_similar} vizinhos; "
                            "não indica citação real)."
                        ),
                    }
                )
    elif include_similarity:
        similarity_skipped = True

    nodes_out: List[Dict[str, Any]] = []
    for a in rows:
        short_label = (a.title or "(sem título)")[:80]
        if len((a.title or "")) > 80:
            short_label += "…"
        group = "unlabeled"
        if a.label == "yes":
            group = "yes"
        elif a.label == "no":
            group = "no"
        abs_full = (a.abstract or "").strip()
        abs_preview = abs_full[:900] + ("…" if len(abs_full) > 900 else "")
        doi_disp = _doi_from_article(a) or ""
        nodes_out.append(
            {
                "id": a.id,
                "label": short_label,
                "fullTitle": a.title or "",
                "year": (getattr(a, "year", None) or "") or "",
                "group": group,
                "doi": doi_disp,
                "abstractPreview": abs_preview,
                "categories": _category_names(a),
                "screeningLabel": _screening_label_pt(a.label),
            }
        )

    return {
        "nodes": nodes_out,
        "edges": citation_edges + sim_edges,
        "stats": {
            "n_articles": len(rows),
            "n_citation_edges": len(citation_edges),
            "n_similarity_edges": len(sim_edges),
            "similarity_skipped": similarity_skipped,
            "citation_hint": _citation_empty_hint(rows, len(citation_edges)),
            "message": (
                f"Ligações de semelhança omitidas: o conjunto tem mais de "
                f"{MAX_SIM_ARTICLES} artigos (limite para o cálculo par a par). "
                "As ligações de citação por DOI continuam a ser mostradas."
                if similarity_skipped
                else None
            ),
        },
    }


def build_similarity_ranking(
    rows: List[_ArticleLike],
    min_similarity: float = DEFAULT_MIN_SIM,
) -> Dict[str, Any]:
    """
    For each article, find the highest Jaccard similarity vs any other row
    (same title+abstract token rule as the graph). Sort descending by that score.
    """
    if not rows:
        return {
            "ranking": [],
            "stats": {"n_articles": 0, "skipped": False, "message": "Não há artigos."},
        }
    if len(rows) > MAX_SIM_ARTICLES:
        return {
            "ranking": [],
            "stats": {
                "n_articles": len(rows),
                "skipped": True,
                "message": (
                    f"Ranking indisponível: o conjunto tem mais de {MAX_SIM_ARTICLES} artigos "
                    "(mesmo limite do cálculo de semelhança no grafo)."
                ),
            },
        }

    tokens = {a.id: _token_set(a.title, a.abstract) for a in rows}
    idx_by_id = {a.id: a for a in rows}

    scored: List[Dict[str, Any]] = []
    for a in rows:
        ta = tokens[a.id]
        best_j = 0.0
        best_id: Optional[int] = None
        for b in rows:
            if b.id == a.id:
                continue
            j = _jaccard(ta, tokens[b.id])
            if j > best_j:
                best_j = j
                best_id = b.id
        best_title = ""
        if best_id is not None:
            best_title = (idx_by_id[best_id].title or "")[:220]

        abs_full = (a.abstract or "").strip()
        abs_preview = abs_full[:900] + ("…" if len(abs_full) > 900 else "")
        group = "unlabeled"
        if a.label == "yes":
            group = "yes"
        elif a.label == "no":
            group = "no"

        scored.append(
            {
                "id": a.id,
                "title": a.title or "",
                "fullTitle": a.title or "",
                "year": (getattr(a, "year", None) or "") or "",
                "group": group,
                "doi": _doi_from_article(a) or "",
                "abstractPreview": abs_preview,
                "categories": _category_names(a),
                "screeningLabel": _screening_label_pt(a.label),
                "best_score": round(best_j, 5),
                "best_pct": round(100 * best_j, 2),
                "best_neighbor_id": best_id,
                "best_neighbor_title": best_title,
                "above_threshold": bool(best_j >= min_similarity),
            }
        )

    scored.sort(key=lambda r: (-float(r["best_score"]), r["id"]))

    return {
        "ranking": scored,
        "stats": {
            "n_articles": len(rows),
            "min_similarity": min_similarity,
            "skipped": False,
            "message": None,
        },
    }
