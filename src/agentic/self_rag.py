from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from configs.config import RAGConfig
from configs.prompt import ANSWER_PROMPT, REWRITE_PROMPT
from src.retrieval.retriever import ProductionRetriever
class RAGState(TypedDict):
    question: str
    original_question: str
    documents: List[str]
    is_relevant: str
    generation: str
    retries: int
class SelfRAGLoopCircuit:
    def __init__(self, retrievers: ProductionRetriever, config: RAGConfig):
        self.retrievers = retrievers
        self.max_retries = 2
        self.llm = ChatOpenAI(model=config.llm_model, temperature=config.temperature)
        self.rewriter_unit = REWRITE_PROMPT | self.llm | StrOutputParser()
        self.grader_unit = ChatPromptTemplate.from_template("검색된 문서가 질문에 답할 수 있는지 'yes'나 'no'로만 출력하세요.\n문서: {context}\n질문: {question}\n판정:") | self.llm | StrOutputParser()
        self.graph = self._compile_fsm_circuit()
    def node_retrieve(self, state: RAGState) -> dict: return {"documents": [d.page_content for d in self.retrievers.retrieve_hybrid(state["question"])[:3]]}
    def node_grade_documents(self, state: RAGState) -> dict: return {"is_relevant": self.grader_unit.invoke({"context": "\n\n".join(state["documents"]), "question": state["original_question"]}).strip().lower()}
    def node_transform_query(self, state: RAGState) -> dict: return {"question": self.rewriter_unit.invoke({"question": state["original_question"]}), "retries": state["retries"] + 1}
    def node_generate(self, state: RAGState) -> dict: return {"generation": self.llm.invoke(ANSWER_PROMPT.format(context="\n\n".join(state["documents"]), question=state["original_question"])).content}
    def condition_decide_path(self, state: RAGState) -> str: return "generate" if "yes" in state["is_relevant"] or state["retries"] >= self.max_retries else "transform_query"
    def _compile_fsm_circuit(self) -> StateGraph:
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self.node_retrieve)
        workflow.add_node("grade_documents", self.node_grade_documents)
        workflow.add_node("transform_query", self.node_transform_query)
        workflow.add_node("generate", self.node_generate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges("grade_documents", self.condition_decide_path, {"generate": "generate", "transform_query": "transform_query"})
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate", END)
        return workflow.compile()
    def run(self, question: str) -> str: return self.graph.invoke({"question": question, "original_question": question, "documents": [], "is_relevant": "", "generation": "", "retries": 0})["generation"]
def build_self_rag_circuit(retrievers: ProductionRetriever, config: RAGConfig) -> SelfRAGLoopCircuit: return SelfRAGLoopCircuit(retrievers, config)
