"""Fast MVP API smoke test; analysis calls are opt-in because they call OpenAI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="also call OpenAI-backed analysis endpoints")
    parser.add_argument("--document", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    report: dict[str, object] = {"checks": [], "live": args.live}
    with httpx.Client(base_url=args.base_url, timeout=20) as client:
        def check(name: str, response, required: tuple[str, ...] = ()):
            body = response.json()
            ok = response.status_code == 200 and all(key in body for key in required)
            report["checks"].append({"name": name, "status": response.status_code, "ok": ok})
            if not ok:
                raise AssertionError(f"{name} failed: {response.status_code} {body}")
            return body

        check("health", client.get("/api/health"), ("status", "chunk_count"))
        documents = check("documents", client.get("/api/documents"))
        if not documents:
            raise AssertionError("document list is empty")
        document_id = args.document or documents[0]["document_id"]
        check("page", client.get("/api/document", params={"document_id": document_id, "page": 1}), ("text", "page_count"))
        check("toc", client.get("/api/toc", params={"document_id": document_id}), ("items",))
        check("search", client.get("/api/search", params={"document_id": document_id, "q": "제출"}), ("results",))
        if args.live:
            for name in ("overview", "risks", "eligibility", "deliverables", "requirements"):
                check(name, client.get(f"/api/analysis/{document_id}/{name}"))
            check("ask", client.post(f"/api/analysis/{document_id}/ask", json={"question": "핵심 요구사항은?", "chat_history": []}))
    output = Path("reports/smoke/latest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"smoke test passed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
