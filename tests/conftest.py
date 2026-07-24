from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from src.observability.wandb import init_wandb_run
from src.search.service import PROJECT_ROOT


_run: Any | None = None
_results: list[dict[str, Any]] = []
_recorded_nodeids: set[str] = set()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("wandb", "W&B test reporting")
    group.addoption(
        "--wandb",
        action="store_true",
        default=False,
        help="Upload pytest module and test results to W&B.",
    )
    group.addoption("--wandb-name", default=None, help="Optional W&B run name.")


def pytest_sessionstart(session: pytest.Session) -> None:
    global _run
    _results.clear()
    _recorded_nodeids.clear()
    if not session.config.getoption("--wandb"):
        return
    _run = init_wandb_run(
        job_type="pytest",
        name=session.config.getoption("--wandb-name"),
        project_root=PROJECT_ROOT,
        config={"test_paths": [str(value) for value in session.config.args]},
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if _run is None or item.nodeid in _recorded_nodeids:
        return

    # Normally the call phase is the final result. Collection/setup failures and
    # skips never reach it, so record those at setup time instead.
    should_record = report.when == "call" or (
        report.when == "setup" and (report.failed or report.skipped)
    )
    if not should_record:
        return
    _recorded_nodeids.add(item.nodeid)
    _results.append({
        "module": item.nodeid.split("::", 1)[0],
        "test": item.nodeid,
        "outcome": report.outcome,
        "duration_seconds": float(report.duration),
        "error": str(report.longrepr) if report.failed else "",
    })


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    global _run
    if _run is None:
        return

    import wandb

    modules: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "skipped": 0, "duration_seconds": 0.0}
    )
    for result in _results:
        metrics = modules[result["module"]]
        metrics[result["outcome"]] += 1
        metrics["duration_seconds"] += result["duration_seconds"]

    table = wandb.Table(
        columns=["module", "test", "outcome", "duration_seconds", "error"],
        data=[
            [row["module"], row["test"], row["outcome"], row["duration_seconds"], row["error"]]
            for row in _results
        ],
    )
    _run.log({
        "pytest/results": table,
        "pytest/total": len(_results),
        "pytest/passed": sum(row["outcome"] == "passed" for row in _results),
        "pytest/failed": sum(row["outcome"] == "failed" for row in _results),
        "pytest/skipped": sum(row["outcome"] == "skipped" for row in _results),
        "pytest/exit_status": int(exitstatus),
    })
    for module, metrics in sorted(modules.items()):
        _run.log({"module": module, **{f"module/{key}": value for key, value in metrics.items()}})
    _run.finish(exit_code=int(exitstatus != pytest.ExitCode.OK))
    _run = None
