# pipeline.py (데이터 흐름 추적용 프로빙 스크립트)
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

    # [LINE 1] Loader 가동
    chunks = load_documents()
    print(f"\n[Probe 1] Loader Out -> Type: {type(chunks)}, Size: {len(chunks)} Chunks")
    print(f"          Sample Payload ID: {chunks[0].metadata['id']}")

    # [LINE 2] Vector DB 가동 (Loader 출력을 입력으로 주입)
    vector_store = build_vector_store(chunks, config)
    print(f"[Probe 2] Vector DB Injected -> Type: {type(vector_store)}")

    # [LINE 3] Retriever 가동 (Vector DB 인스턴스를 입력으로 주입)
    retrievers = build_retrievers(chunks, vector_store, config)
    print("[Probe 3] Retrievers Wired -> Naive/Hybrid Units Initialized.")

    # [LINE 4] Chain 조립 (Retriever 인스턴스를 입력으로 주입)
    advanced_chain = build_advanced_chain(retrievers, config)
    print("[Probe 4] Control Chain Assembled. Ready for Input Signal.")

    # [LINE 5] 최종 계측
    run_evaluation(advanced_chain, retrievers, config)


if __name__ == "__main__":
    main()
