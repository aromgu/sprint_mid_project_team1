from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv

from configs.config import RAGConfig
from src.embeddings.build_embeddings import build_vector_store
from src.evaluation.eval_rag import run_evaluation
from src.generation.generate_answer import build_advanced_chain
from src.loader.load_documents import load_documents
from src.retrieval.retriever import build_retrievers


def main():
    load_dotenv()
    config = RAGConfig()
    chunks = load_documents()
    vector_store = build_vector_store(chunks, config)
    retrievers = build_retrievers(chunks, vector_store, config)
    chain = build_advanced_chain(retrievers, config)
    run_evaluation(chain, retrievers, config)
    print("Evaluation completed.")


if __name__ == "__main__":
    main()
