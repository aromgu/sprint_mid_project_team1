from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dotenv import load_dotenv

from configs.config import RAGConfig
from src.embeddings.build_embeddings import build_vector_store
from src.loader.load_documents import load_documents


def main():
    load_dotenv()
    config = RAGConfig()
    chunks = load_documents()
    build_vector_store(chunks, config)
    print("Indexing completed.")


if __name__ == "__main__":
    main()
