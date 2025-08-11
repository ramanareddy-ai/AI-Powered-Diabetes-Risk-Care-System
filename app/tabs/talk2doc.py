"""
Talk to Doctor page for the Diabetes Risk & Care System.

This module defines a simple interface where users can ask questions
about diabetes and receive AI‑generated answers. It uses the language
model wrapper defined in :mod:`app.core.llm`.
"""

from __future__ import annotations

import streamlit as st

from app.core.llm import chat


def render() -> None:
    """Render the chat interface."""
    st.header("Talk to a Virtual Doctor")
    st.write(
        "Ask any question about diabetes. The AI will provide a general answer. "
        "Please remember that this does not constitute medical advice."
    )
    question = st.text_area("Your question:")
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = chat(
                        question,
                        system="You are a compassionate physician. Provide concise, clear, and empathetic answers.",
                    )
                except Exception as exc:
                    answer = (
                        "Unable to answer your question because the OpenAI API key "
                        f"is not configured. Exception: {exc}"
                    )
            st.markdown(answer)
