"""LangGraph node implementations for the curriculum RAG agent.

Graph flow:
    START -> agent -> [route] -> tools -> grade_documents -> agent (loop)
                   |-> (no tools) -> generate -> END

The agent can loop through tools multiple times to gather all needed
information before producing a final answer.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from src.chain.prompts import (
    AGENT_SYSTEM_PROMPT,
    GENERATE_SYSTEM_PROMPT,
    GRADING_PROMPT,
    REWRITE_PROMPT,
)
from src.chain.state import AgentState
from src.config import settings
from src.retrieval.tools import ALL_TOOLS

MAX_RETRIES = 1
MAX_TOOL_CALLS = 6


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        openai_api_key=settings.openai_api_key,
    )


def _get_last_human_question(state: AgentState) -> str:
    """Extract the most recent human question from state messages."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _last_tool_name(state: AgentState) -> str | None:
    """Get the tool name from the most recent AI tool call."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return msg.tool_calls[-1]["name"]
    return None


def _extract_current_turn(messages: list) -> list:
    """Extract only the current turn's messages for generation.

    Walks backwards from the end to find the last HumanMessage (the current
    question), then returns everything from that point forward. This prevents
    prior turns' answers from bleeding into the new response.
    """
    last_human_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break

    if last_human_idx is None:
        return messages

    return messages[last_human_idx:]


def _build_system_prompt(template: str, state: AgentState) -> str:
    """Fill in the student_context placeholder in a prompt template."""
    ctx = state.get("student_context", "")
    return template.format(student_context=ctx)


def agent_node(state: AgentState) -> dict:
    """Invoke the LLM with tools bound. It decides to call a tool or respond."""
    tool_call_count = state.get("tool_call_count", 0)

    if tool_call_count >= MAX_TOOL_CALLS:
        logger.warning(f"Tool call limit reached ({MAX_TOOL_CALLS}), forcing direct response")
        llm = _get_llm()
    else:
        llm = _get_llm().bind_tools(ALL_TOOLS)

    messages = state["messages"]

    if not messages or not isinstance(messages[0], SystemMessage):
        prompt = _build_system_prompt(AGENT_SYSTEM_PROMPT, state)
        messages = [SystemMessage(content=prompt)] + list(messages)

    response = llm.invoke(messages)
    return {"messages": [response]}


def grade_documents(state: AgentState) -> dict:
    """Grade retrieved documents for relevance.

    Structured tools (prerequisite_lookup, combo_navigator, curriculum_browser)
    skip grading since they return deterministic data. Only vector_search
    results are graded. Increments tool_call_count as a loop safety counter.
    """
    tool_name = _last_tool_name(state)
    retry_count = state.get("retry_count", 0)
    tool_call_count = state.get("tool_call_count", 0) + 1

    if tool_name != "vector_search":
        logger.debug(f"Skipping grading for structured tool: {tool_name}")
        return {
            "grading_decision": "relevant",
            "retry_count": retry_count,
            "tool_call_count": tool_call_count,
        }

    question = _get_last_human_question(state)
    last_msg = state["messages"][-1]
    documents = last_msg.content if hasattr(last_msg, "content") else ""

    llm = _get_llm()

    grading_input = GRADING_PROMPT.format(question=question, documents=documents)
    grade_response = llm.invoke([HumanMessage(content=grading_input)])
    grade = grade_response.content.strip().lower()

    if "yes" in grade:
        logger.info("Document grading: RELEVANT")
        return {
            "grading_decision": "relevant",
            "retry_count": retry_count,
            "tool_call_count": tool_call_count,
        }

    if retry_count < MAX_RETRIES:
        logger.info("Document grading: NOT RELEVANT, rewriting query")
        rewrite_input = REWRITE_PROMPT.format(question=question)
        rewrite_response = llm.invoke([HumanMessage(content=rewrite_input)])
        rewritten = rewrite_response.content.strip()

        return {
            "messages": [HumanMessage(content=rewritten)],
            "grading_decision": "rewrite",
            "retry_count": retry_count + 1,
            "tool_call_count": tool_call_count,
        }

    logger.info("Document grading: NOT RELEVANT, max retries reached")
    return {
        "grading_decision": "not_relevant",
        "retry_count": retry_count,
        "tool_call_count": tool_call_count,
    }


def generate(state: AgentState) -> dict:
    """Produce the final answer from the current turn's tool results only.

    Only passes the current question + its tool calls/results to the LLM,
    so prior conversation turns don't bleed into the answer.
    """
    llm = _get_llm()
    prompt = _build_system_prompt(GENERATE_SYSTEM_PROMPT, state)
    current_turn = _extract_current_turn(state["messages"])
    messages = [SystemMessage(content=prompt)] + current_turn
    response = llm.invoke(messages)
    return {"messages": [response]}
