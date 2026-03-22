"""Streamlit chat interface for the FPT University curriculum advisor."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.chain.agent import AgentResponse, chat
from src.profiles import PROFILES, StudentProfile

st.set_page_config(
    page_title="FPT Curriculum Advisor",
    layout="wide",
)

with st.sidebar:
    st.title("FPT Curriculum Advisor")
    st.caption("BIT_AI Program — K20-K21")

    st.subheader("Student profile")
    profile_key = st.selectbox(
        "Select a test profile",
        options=list(PROFILES.keys()),
        format_func=lambda k: f"{PROFILES[k].name} — {PROFILES[k].description}",
        key="profile_selector",
    )
    active_profile: StudentProfile = PROFILES[profile_key]

    if active_profile.current_semester > 0:
        with st.expander("Profile details"):
            st.text(active_profile.summary())

    st.divider()

    st.markdown(
        "**What you can ask:**\n"
        "- What courses are in semester X?\n"
        "- What are the prerequisites for [course]?\n"
        "- What does [course] cover?\n"
        "- What combos/specializations are available?\n"
        "- How do I get to [course] as fast as possible?\n"
        "- What courses does completing [course] unlock?\n"
        "- Should I retake [course] to improve my GPA?\n"
    )
    st.divider()

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.citations = []
        st.rerun()

    if st.session_state.get("citations"):
        st.subheader("Sources")
        for cite in st.session_state.citations:
            tool_name = cite["name"]
            args = cite["args"]
            args_str = ", ".join(f"{k}={v!r}" for k, v in args.items() if v)
            with st.expander(f"{tool_name}({args_str})", expanded=False):
                result_text = cite["result"]
                if len(result_text) > 1000:
                    st.code(result_text[:1000] + "\n... (truncated)", language="markdown")
                else:
                    st.code(result_text, language="markdown")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "citations" not in st.session_state:
    st.session_state.citations = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about your curriculum..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response: AgentResponse = chat(
                question=prompt,
                history=st.session_state.messages[:-1],
                profile=active_profile,
            )
        st.markdown(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.session_state.citations = response.tool_calls
    st.rerun()
