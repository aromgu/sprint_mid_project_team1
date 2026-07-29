"""Workspace-relative configuration for the ported Main Advanced RAG."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/main_advanced_rag.yaml")


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


@dataclass(frozen=True)
class MainRAGSettings:
    config_path: Path
    project_root: Path
    values: dict[str, Any]

    def path(self, name: str) -> Path:
        try:
            value = self.values["paths"][name]
        except KeyError as exc:
            raise KeyError(f"main Advanced RAG 경로 설정이 없습니다: paths.{name}") from exc
        return _resolve(self.project_root, str(value))

    def get(self, section: str, name: str, default: Any = None) -> Any:
        return self.values.get(section, {}).get(name, default)


def load_settings(path: str | Path = DEFAULT_CONFIG_PATH) -> MainRAGSettings:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Main Advanced RAG 설정 파일이 없습니다: {config_path}")
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("Main Advanced RAG 설정은 YAML 객체여야 합니다")
    # configs/<file>.yaml is one level below the repository root.
    project_root = config_path.parent.parent
    return MainRAGSettings(config_path, project_root, values)
