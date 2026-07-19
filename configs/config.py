import os
class RAGConfig:
    def __init__(self):
        self.llm_model = "gpt-4o-mini"
        self.embedding_model = "text-embedding-3-small"
        self.temperature = 0
        self.naive_k = 3
        self.wide_k = 8
        self.rerank_top_n = 3
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.report_dir = os.path.join(self.base_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)
