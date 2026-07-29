from __future__ import annotations

import json
import hashlib
import re
import asyncio
from pathlib import Path
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request

from backend.models import DocumentSummary
from src.chunking.structured_chunker import chunk_pages
from src.ingestion.models import DocumentManifest
from src.ingestion.pdf_parser import parse_document
from src.ingestion.manifest import write_manifest

router = APIRouter(prefix="/api", tags=["documents"])


def meaningful_heading(value: str | None) -> bool:
    """Exclude PDF parser placeholders and page labels from the visible TOC."""
    title = re.sub(r"\s+", " ", value or "").strip()
    compact = re.sub(r"[\s._-]+", "", title).casefold()
    if not compact or compact in {"본문", "목차", "차례", "contents", "tableofcontents"}:
        return False
    if re.fullmatch(r"(?:p(?:age)?\.?\s*)?\d+\s*(?:쪽|페이지)?", title, re.IGNORECASE):
        return False
    if re.fullmatch(r"[-–—_=·•.\s]+", title):
        return False
    return bool(re.search(r"[A-Za-z가-힣]", title))


def manifest() -> dict:
    path = Path(__file__).resolve().parents[2] / "data/manifests/documents.json"
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def pages() -> list[dict]:
    path = Path(__file__).resolve().parents[2] / "data/processed/pages.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@router.get("/documents", response_model=list[DocumentSummary])
def documents():
    rows = []
    for item in manifest()["documents"]:
        pdf_path = Path(item.get("pdf_path", ""))
        document_date = None
        if pdf_path.is_file():
            document_date = pdf_path.stat().st_mtime_ns
        rows.append(DocumentSummary(
            document_id=item["document_id"], title=item["title"], organization=item["organization"],
            difficulty=item.get("difficulty"), document_date=str(document_date) if document_date else None,
        ))
    return rows


@router.put("/documents/upload")
async def upload_document(request: Request, filename: str = "uploaded.pdf", title: str | None = None, organization: str = "사용자 업로드"):
    """Ingest a PDF sent as the raw request body and rebuild the in-process search service."""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="filename must end with .pdf")
    content = await request.body()
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="request body is not a PDF")
    root = Path(__file__).resolve().parents[2]
    upload_dir = root / "data" / "uploads"; upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    path = upload_dir / safe_name; path.write_bytes(content)
    document_id = f"upload_{hashlib.sha1(content).hexdigest()[:10]}"
    manifest_item = DocumentManifest(document_id=document_id, difficulty="medium", organization=organization,
        title=title or Path(safe_name).stem, csv_filename=safe_name, pdf_path=str(path), filename=safe_name,
        file_size=len(content), sha256=hashlib.sha256(content).hexdigest(), match_score=1.0,
        target_accuracy="pending", complexity="uploaded")
    from src.main_rag.upload_indexer import index_uploaded_pdf
    advanced = await asyncio.to_thread(
        index_uploaded_pdf,
        path,
        document_id=document_id,
        title=manifest_item.title,
        organization=organization,
        retriever=request.app.state.rag_client.advanced_retriever,
    )
    pages_out, _ = parse_document(manifest_item)
    chunks_out = chunk_pages(pages_out)
    pages_path = root / "data/processed/pages.jsonl"; chunks_path = root / "data/processed/chunks.jsonl"
    with pages_path.open("a", encoding="utf-8") as handle:
        for item in pages_out: handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    with chunks_path.open("a", encoding="utf-8") as handle:
        for item in chunks_out: handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    payload = manifest(); payload["documents"].append(manifest_item.to_dict()); payload["document_count"] = len(payload["documents"])
    write_manifest([DocumentManifest(**item) for item in payload["documents"]], root / "data/manifests/documents.json")
    pages.cache_clear()
    from src.search.service import SearchService
    from backend.services.rag_client import RAGClient
    request.app.state.search_service = SearchService()
    request.app.state.rag_client = RAGClient(search_service=request.app.state.search_service)
    return {
        "document_id": document_id, "title": manifest_item.title,
        "pages": len(pages_out), "chunks": len(chunks_out), **advanced,
    }


@router.get("/documents/{document_id}")
def document(document_id: str):
    item = next((item for item in manifest()["documents"] if item["document_id"] == document_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="document not found")
    return item


@router.get("/document")
def page(document_id: str, page: int):
    item = next((item for item in pages() if item["document_id"] == document_id and item["page"] == page), None)
    if item is None:
        raise HTTPException(status_code=404, detail="page not found")
    page_count = sum(1 for row in pages() if row["document_id"] == document_id)
    return {"document_id": document_id, "page": page, "page_count": page_count, "text": item["text"], "headings": item.get("headings", []), "image_count": item.get("image_count", 0), "ocr_applied": item.get("ocr_applied", False)}


@router.get("/toc")
def toc(document_id: str):
    entries = []
    seen = set()
    for item in pages():
        if item["document_id"] != document_id:
            continue
        for heading in item.get("headings", []):
            if not meaningful_heading(heading):
                continue
            heading = re.sub(r"\s+", " ", heading).strip()
            key = (heading, item["page"])
            if key not in seen:
                entries.append({"id": f"{document_id}-{item['page']}-{len(entries)}", "title": heading, "page": item["page"]})
                seen.add(key)
    return {"document_id": document_id, "items": entries}


@router.get("/search")
def search(document_id: str, q: str, top_k: int = 5):
    if not q.strip():
        return {"document_id": document_id, "results": []}
    query = q.casefold().strip()
    # PDF text commonly contains line breaks or punctuation inside a phrase.
    # Search individual terms as well as the exact phrase so normal multi-word
    # queries still return useful source pages.
    terms = list(dict.fromkeys(re.findall(r"[0-9a-z가-힣]+", query)))
    if not terms:
        return {"document_id": document_id, "results": []}
    results = []
    for item in pages():
        if item["document_id"] != document_id:
            continue
        text = item["text"]
        folded = text.casefold()
        counts = {term: folded.count(term) for term in terms}
        matched = [term for term, count in counts.items() if count]
        if matched:
            exact_bonus = 2 if query in folded else 0
            coverage = len(matched) / len(terms)
            score = round(coverage * 10 + sum(counts.values()) + exact_bonus, 3)
            positions = [folded.find(term) for term in matched]
            index = min(position for position in positions if position >= 0)
            excerpt = text[max(0, index - 100):index + 280].replace("\n", " ")
            results.append({
                "page": item["page"], "excerpt": excerpt, "score": score,
                "matched_terms": matched, "match_count": len(matched),
            })
    results.sort(key=lambda row: (-row["match_count"], -row["score"], row["page"]))
    return {"document_id": document_id, "results": results[:max(1, min(top_k, 20))]}


@router.get("/health")
def health(request: Request):
    service = request.app.state.search_service
    return {
        "status": "ok", "chunk_count": len(service.chunks),
        "default_retriever": "main_advanced_dense",
        "available_retrievers": ["main_advanced_dense"],
        "conversation_sessions": request.app.state.rag_client.sessions.session_count,
    }
