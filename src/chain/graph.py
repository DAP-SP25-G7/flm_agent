"""LangGraph graph assembly for the curriculum RAG agent.

Graph structure:
    START -> agent -> [route] -> tools -> grade_documents -> agent (loop)
                   |-> (no tools) -> generate -> END

The agent can call tools multiple times in a loop. After each tool call,
grade_documents checks relevance and routes back to agent, which decides
whether to call more tools or respond. When agent responds without tool
calls, it goes to generate for final synthesis.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.chain.nodes import agent_node, generate, grade_documents
from src.chain.state import AgentState
from src.retrieval.tools import ALL_TOOLS


def _route_after_agent(state: AgentState) -> str:
    """Route after agent: tools if tool_calls, generate if direct response."""
    result = tools_condition(state)
    if result == END:
        return "generate"
    return "tools"


def build_graph() -> StateGraph:
    """Construct the curriculum advisor agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("generate", generate)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "generate": "generate"},
    )
    graph.add_edge("tools", "grade_documents")
    graph.add_edge("grade_documents", "agent")
    graph.add_edge("generate", END)

    return graph


def get_compiled_graph():
    """Build and compile the graph, ready for invocation."""
    graph = build_graph()
    return graph.compile()
