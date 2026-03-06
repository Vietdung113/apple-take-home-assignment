"""LangGraph extractive-first pipeline with chunking.

Flow: chunk_document → extract_facts (parallel) → merge_facts → summarize_facts → finalize
"""

from langgraph.graph import StateGraph, START, END

from api_service.agents.state import SummaryState
from api_service.agents.nodes.chunk_document import chunk_document_node
from api_service.agents.nodes.extract_facts import extract_facts_node
from api_service.agents.nodes.merge_facts import merge_facts_node
from api_service.agents.nodes.summarize_facts import summarize_facts_node


def finalize(state: SummaryState) -> dict:
    """Copy draft to final_summary."""
    return {"final_summary": state["draft_summary"]}


def build_graph() -> StateGraph:
    graph = StateGraph(SummaryState)

    graph.add_node("chunk_document", chunk_document_node)
    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("merge_facts", merge_facts_node)
    graph.add_node("summarize_facts", summarize_facts_node)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "chunk_document")
    graph.add_edge("chunk_document", "extract_facts")
    graph.add_edge("extract_facts", "merge_facts")
    graph.add_edge("merge_facts", "summarize_facts")
    graph.add_edge("summarize_facts", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


pipeline = build_graph()
