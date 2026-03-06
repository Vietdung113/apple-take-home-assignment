"""LangGraph adaptive summarization pipeline.

Flow:
- Short docs (<15K tokens): chunk_document → direct_summarize → finalize
- Long docs (≥15K tokens): chunk_document → extract_facts → merge_facts → summarize_facts → finalize
"""

from langgraph.graph import StateGraph, START, END

from api_service.agents.state import SummaryState
from api_service.agents.nodes.chunk_document import chunk_document_node
from api_service.agents.nodes.direct_summarize import direct_summarize_node
from api_service.agents.nodes.extract_facts import extract_facts_node
from api_service.agents.nodes.merge_facts import merge_facts_node
from api_service.agents.nodes.summarize_facts import summarize_facts_node


def finalize(state: SummaryState) -> dict:
    """Copy draft to final_summary."""
    return {"final_summary": state["draft_summary"]}


def route_after_chunk(state: SummaryState) -> str:
    """Route based on document length: direct or extract path."""
    if state["is_long_document"]:
        return "extract_facts"
    else:
        return "direct_summarize"


def build_graph() -> StateGraph:
    graph = StateGraph(SummaryState)

    # Add all nodes
    graph.add_node("chunk_document", chunk_document_node)
    graph.add_node("direct_summarize", direct_summarize_node)
    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("merge_facts", merge_facts_node)
    graph.add_node("summarize_facts", summarize_facts_node)
    graph.add_node("finalize", finalize)

    # Start → routing node
    graph.add_edge(START, "chunk_document")

    # Conditional routing after chunk_document
    graph.add_conditional_edges(
        "chunk_document",
        route_after_chunk,
        {
            "direct_summarize": "direct_summarize",
            "extract_facts": "extract_facts",
        }
    )

    # Direct path: short docs
    graph.add_edge("direct_summarize", "finalize")

    # Extract path: long docs
    graph.add_edge("extract_facts", "merge_facts")
    graph.add_edge("merge_facts", "summarize_facts")
    graph.add_edge("summarize_facts", "finalize")

    # End
    graph.add_edge("finalize", END)

    return graph.compile()


pipeline = build_graph()
