"""LangGraph adaptive summarization pipeline.

Flow (optimized based on GovReport analysis):
- Short docs (<100K chars / 25K tokens): chunk_document → direct_summarize → finalize
- Long docs (≥100K chars): chunk_document → summarize_chunks (hierarchical 2-level) → finalize

Rationale:
- 97% of GovReport docs are <20K tokens → direct path optimal
- Only 3% need hierarchical approach
- Qwen3-0.6B handles 32K context well

Hierarchical strategy:
- Level 1: 3-5 chunks (~15K tokens each) → section summaries (parallel)
- Level 2: Merge section summaries → final summary
"""

from langgraph.graph import StateGraph, START, END

from api_service.agents.state import SummaryState
from api_service.agents.nodes.chunk_document import chunk_document_node
from api_service.agents.nodes.direct_summarize import direct_summarize_node
from api_service.agents.nodes.summarize_chunks import summarize_chunks_node


def finalize(state: SummaryState) -> dict:
    """Copy draft to final_summary."""
    return {"final_summary": state["draft_summary"]}


def route_after_chunk(state: SummaryState) -> str:
    """Route based on document length: direct or hierarchical path.

    Threshold: 100K chars (~25K tokens) based on GovReport analysis:
    - 97% of docs fit direct path (< 20K tokens)
    - Direct fine-tuned inference often outperforms agent for short-medium docs
    - Agent only adds value for longest 3% of documents
    """
    doc_length = len(state["document"])
    AGENT_THRESHOLD = 120000

    if doc_length >= AGENT_THRESHOLD:
        return "summarize_chunks"  # Hierarchical for very long docs
    else:
        return "direct_summarize"  # Direct for 97% of cases


def build_graph() -> StateGraph:
    graph = StateGraph(SummaryState)

    # Add all nodes
    graph.add_node("chunk_document", chunk_document_node)
    graph.add_node("direct_summarize", direct_summarize_node)
    graph.add_node("summarize_chunks", summarize_chunks_node)
    graph.add_node("finalize", finalize)

    # Start → routing node
    graph.add_edge(START, "chunk_document")

    # Conditional routing after chunk_document
    graph.add_conditional_edges(
        "chunk_document",
        route_after_chunk,
        {
            "direct_summarize": "direct_summarize",
            "summarize_chunks": "summarize_chunks",
        }
    )

    # Direct path: short docs
    graph.add_edge("direct_summarize", "finalize")

    # Hierarchical path: long docs
    graph.add_edge("summarize_chunks", "finalize")

    # End
    graph.add_edge("finalize", END)

    return graph.compile()


pipeline = build_graph()
