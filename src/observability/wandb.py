from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def init_wandb_run(
    *,
    job_type: str,
    config: dict[str, Any] | None = None,
    name: str | None = None,
    project_root: Path | None = None,
):
    """Initialize W&B using local env files without ever handling keys in code."""
    if project_root:
        load_dotenv(project_root / ".env", override=False)
        load_dotenv(project_root / ".env.local", override=True)
    if not os.getenv("WANDB_API_KEY") and os.getenv("WANDB_MODE", "online") != "offline":
        raise RuntimeError(
            "WANDB_API_KEY is not configured. Add it to .env.local or run `uv run wandb login`."
        )

    import wandb
    import weave

    project = os.getenv("WANDB_PROJECT", "rfp-action-copilot-local")
    entity = os.getenv("WANDB_ENTITY") or None
    # Weave provides call-level LLM traces while wandb.Run stores aggregate
    # experiment metrics. Initializing both with the same entity/project links them.
    weave.init(f"{entity}/{project}" if entity else project)
    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        job_type=job_type,
        tags=[tag.strip() for tag in os.getenv("WANDB_TAGS", "").split(",") if tag.strip()],
        config=config or {},
    )
