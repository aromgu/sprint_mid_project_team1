from typing import List

from langchain_core.documents import Document


def split_text(documents: List[Document], chunk_size: int = 500, overlap: int = 0) -> List[Document]:
    if chunk_size <= 0:
        return documents

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    return documents
