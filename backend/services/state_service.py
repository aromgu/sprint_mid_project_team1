from __future__ import annotations

import json
from pathlib import Path
from threading import Lock


class UserStateService:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = Lock()

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def update(self, document_id: str, section: str, item_id: str, values: dict) -> dict:
        with self.lock:
            payload = self._read()
            document = payload.setdefault(document_id, {})
            items = document.setdefault(section, {})
            items.setdefault(item_id, {}).update(values)
            self._write(payload)
            return items[item_id]

    def get(self, document_id: str) -> dict:
        with self.lock:
            return self._read().get(document_id, {})
