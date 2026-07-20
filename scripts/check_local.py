"""Run local project checks before committing or opening a pull request.

The default checks are deterministic and do not call external APIs. Use
``--pipeline`` explicitly when a full OpenAI-backed pipeline run is required.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


Check = tuple[str, Callable[[], str]]


def check_python_syntax() -> str:
    """Compile project Python sources in memory without creating cache files."""
    roots = [ROOT / "configs", ROOT / "scripts", ROOT / "src"]
    files = [ROOT / "pipeline.py"]

    for source_root in roots:
        if source_root.exists():
            files.extend(source_root.rglob("*.py"))

    excluded_parts = {".venv", "__pycache__", "TMP", "agentic"}
    checked = 0
    for path in sorted(set(files)):
        if excluded_parts.intersection(path.parts):
            continue
        source = path.read_text(encoding="utf-8-sig")
        compile(source, str(path), "exec")
        checked += 1

    if checked == 0:
        raise AssertionError("no Python files were found")
    return f"{checked} Python files"


def check_core_imports() -> str:
    """Import the public pipeline components without invoking an API."""
    from configs.config import RAGConfig
    from src.embeddings.build_embeddings import build_vector_store
    from src.evaluation.eval_rag import run_evaluation
    from src.generation.generate_answer import build_advanced_chain
    from src.loader.load_documents import load_documents
    from src.retrieval.retriever import build_retrievers

    symbols = {
        "RAGConfig": RAGConfig,
        "load_documents": load_documents,
        "build_vector_store": build_vector_store,
        "build_retrievers": build_retrievers,
        "build_advanced_chain": build_advanced_chain,
        "run_evaluation": run_evaluation,
    }
    invalid = [name for name, symbol in symbols.items() if not callable(symbol)]
    if invalid:
        raise AssertionError(f"not callable: {', '.join(invalid)}")
    return f"{len(symbols)} core symbols"


def check_interfaces() -> str:
    """Check the function contracts used by pipeline.py."""
    from src.embeddings.build_embeddings import build_vector_store
    from src.evaluation.eval_rag import run_evaluation
    from src.generation.generate_answer import build_advanced_chain
    from src.loader.load_documents import load_documents
    from src.retrieval.retriever import build_retrievers

    expected = {
        load_documents: [],
        build_vector_store: ["chunks", "config"],
        build_retrievers: ["chunks", "vector_store", "config"],
        build_advanced_chain: ["retrievers", "config"],
        run_evaluation: ["advanced_chain", "retrievers", "config"],
    }

    for function, parameter_names in expected.items():
        actual = list(inspect.signature(function).parameters)
        if actual != parameter_names:
            raise AssertionError(
                f"{function.__name__}{tuple(actual)} does not match "
                f"expected parameters {tuple(parameter_names)}"
            )
    return f"{len(expected)} interfaces"


def check_config() -> str:
    from configs.config import RAGConfig

    config = RAGConfig()
    if not config.llm_model:
        raise AssertionError("llm_model is empty")
    if not config.embedding_model:
        raise AssertionError("embedding_model is empty")
    for name in ("naive_k", "wide_k", "rerank_top_n"):
        value = getattr(config, name)
        if not isinstance(value, int) or value <= 0:
            raise AssertionError(f"{name} must be a positive integer")
    if config.wide_k < config.rerank_top_n:
        raise AssertionError("wide_k must be greater than or equal to rerank_top_n")

    report_dir = Path(config.report_dir).resolve()
    expected_report_dir = ROOT / "reports"
    if report_dir != expected_report_dir.resolve():
        raise AssertionError(f"unexpected report_dir: {report_dir}")
    if not report_dir.is_dir():
        raise AssertionError(f"report directory does not exist: {report_dir}")
    return f"models configured, reports={report_dir.relative_to(ROOT)}"


def check_loader_contract() -> str:
    from langchain_core.documents import Document

    from src.loader.load_documents import load_documents

    documents = load_documents()
    if not isinstance(documents, list) or not documents:
        raise AssertionError("load_documents() must return a non-empty list")

    document_ids: list[str] = []
    for index, document in enumerate(documents):
        if not isinstance(document, Document):
            raise AssertionError(f"item {index} is not a Document")
        if not document.page_content.strip():
            raise AssertionError(f"item {index} has empty page_content")
        for metadata_key in ("id", "source"):
            if not document.metadata.get(metadata_key):
                raise AssertionError(f"item {index} is missing metadata.{metadata_key}")
        document_ids.append(str(document.metadata["id"]))

    if len(document_ids) != len(set(document_ids)):
        raise AssertionError("document metadata.id values must be unique")
    return f"{len(documents)} documents with id/source metadata"


def check_text_processing() -> str:
    from langchain_core.documents import Document

    from src.chunking.split_text import split_text
    from src.preprocessing.clean_text import clean_text

    cleaned = clean_text("  문서\n\n  내용   확인  ")
    if cleaned != "문서 내용 확인":
        raise AssertionError(f"unexpected clean_text result: {cleaned!r}")

    documents = [Document(page_content="test", metadata={"id": "test"})]
    chunks = split_text(documents, chunk_size=500, overlap=0)
    if not isinstance(chunks, list) or not chunks:
        raise AssertionError("split_text() must return a non-empty list")

    try:
        split_text(documents, chunk_size=10, overlap=10)
    except ValueError:
        pass
    else:
        raise AssertionError("split_text() must reject overlap >= chunk_size")
    return "cleaning and chunking contracts"


def check_evaluation_dataset() -> str:
    from src.dataset import get_hitk_testset, get_ragas_testset

    hitk = get_hitk_testset()
    if not hitk:
        raise AssertionError("Hit@K test set is empty")
    for index, row in enumerate(hitk):
        if not isinstance(row, tuple) or len(row) != 3 or not all(row):
            raise AssertionError(f"invalid Hit@K row at index {index}: {row!r}")

    ragas = get_ragas_testset()
    if set(ragas) != {"questions", "ground_truths"}:
        raise AssertionError("RAGAS test set must contain questions and ground_truths")
    if not ragas["questions"]:
        raise AssertionError("RAGAS questions are empty")
    if len(ragas["questions"]) != len(ragas["ground_truths"]):
        raise AssertionError("RAGAS questions and ground_truths have different lengths")
    return f"Hit@K={len(hitk)}, RAGAS={len(ragas['questions'])}"


def check_environment_file() -> str:
    """Check only the variable name; never read or print the API key value."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return ".env absent (required only for API-backed runs)"

    variable_names: set[str] = set()
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        variable_names.add(line.split("=", 1)[0].removeprefix("export ").strip())
    if "OPENAI_API_KEY" not in variable_names:
        raise AssertionError(".env does not define OPENAI_API_KEY")
    return ".env defines OPENAI_API_KEY (value hidden)"


def run_checks() -> bool:
    checks: list[Check] = [
        ("Python syntax", check_python_syntax),
        ("Core imports", check_core_imports),
        ("Pipeline interfaces", check_interfaces),
        ("Configuration", check_config),
        ("Loader contract", check_loader_contract),
        ("Text processing", check_text_processing),
        ("Evaluation dataset", check_evaluation_dataset),
        ("Environment", check_environment_file),
    ]

    failures: list[tuple[str, BaseException]] = []
    for name, check in checks:
        try:
            detail = check()
        except Exception as error:  # Continue to report every failed check.
            failures.append((name, error))
            print(f"[FAIL] {name}: {type(error).__name__}: {error}")
        else:
            print(f"[PASS] {name}: {detail}")

    print()
    if failures:
        print(
            f"Local checks failed: {len(checks) - len(failures)}/{len(checks)} passed"
        )
        return False
    print(f"Local checks passed: {len(checks)}/{len(checks)}")
    return True


def run_pipeline() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for --pipeline")

    from pipeline import main

    print("\nRunning API-backed pipeline check...")
    main()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run API-free checks (currently the same as the default)",
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="run pipeline.py after local checks; this calls the OpenAI API",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not run_checks():
        return 1
    if args.pipeline:
        run_pipeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
