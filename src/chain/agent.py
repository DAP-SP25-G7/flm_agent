"""Public API for the curriculum RAG agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.chain.graph import get_compiled_graph
from src.profiles import StudentProfile


@dataclass
class AgentResponse:
    """Structured response from the agent, including citations."""

    content: str
    tool_calls: list[dict] = field(default_factory=list)


def _extract_citations(messages: list) -> list[dict]:
    """Walk through messages and pair each tool call with its result."""
    citations: list[dict] = []
    pending_calls: dict[str, dict] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                pending_calls[tc["id"]] = {
                    "name": tc["name"],
                    "args": tc["args"],
                }
        elif isinstance(msg, ToolMessage):
            call_id = msg.tool_call_id
            if call_id in pending_calls:
                info = pending_calls.pop(call_id)
                citations.append({
                    "name": info["name"],
                    "args": info["args"],
                    "result": msg.content,
                })

    return citations


def chat(
    question: str,
    history: list[dict] | None = None,
    profile: StudentProfile | None = None,
) -> AgentResponse:
    """Send a question to the curriculum advisor agent.

    Args:
        question: The user's question.
        history: Optional conversation history as a list of
                 {"role": "user"|"assistant", "content": str} dicts.
        profile: Optional student profile for personalized answers.

    Returns:
        AgentResponse with content and tool call citations.
    """
    graph = get_compiled_graph()

    messages: list = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

    student_context = ""
    if profile and profile.current_semester > 0:
        student_context = (
            "**Current student context (use this to personalize your answers):**\n"
            + profile.summary()
        )

    messages.append(HumanMessage(content=question))

    result = graph.invoke({
        "messages": messages,
        "retry_count": 0,
        "tool_call_count": 0,
        "grading_decision": "",
        "student_context": student_context,
    })

    final_messages = result.get("messages", [])

    answer = "I wasn't able to generate a response. Please try rephrasing your question."
    for msg in reversed(final_messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            answer = msg.content
            break

    citations = _extract_citations(final_messages)

    return AgentResponse(content=answer, tool_calls=citations)
