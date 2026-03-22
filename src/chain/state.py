"""Agent state definition for the LangGraph curriculum advisor."""

from __future__ import annotations

from typing import Annotated

from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """Extended state for the curriculum RAG agent.

    Inherits ``messages`` (with add-reducer) from MessagesState.
    """

    retry_count: int
    tool_call_count: int
    grading_decision: str
    student_context: str
