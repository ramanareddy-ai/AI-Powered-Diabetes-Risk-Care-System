"""
Utility functions to interact with the OpenAI API.

This module defines a simple wrapper around the OpenAI Chat API that
automatically reads the API key from either the environment variable
``OPENAI_API_KEY`` or Streamlit secrets. If no key is configured, an
exception is raised to alert the caller.

Functions in this module hide the complexity of instantiating the client and
performing calls, making it easy for the rest of the application to use a
language model for generating human‑friendly explanations.
"""

from __future__ import annotations

"""
Human-friendly OpenAI wrapper.

- Reads the API key from env var or Streamlit secrets.
- Provides a simple `chat()` function that accepts `system` and `temperature`.
"""
from typing import Optional, List, Dict
import os
import streamlit as st
from openai import OpenAI

_client: Optional[OpenAI] = None


def _get_api_key() -> Optional[str]:
    """
    Prefer environment variable; fall back to Streamlit secrets.
    """
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None


def _get_client() -> OpenAI:
    """
    Create (or reuse) the OpenAI client.
    """
    global _client
    if _client is None:
        api_key = _get_api_key()
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. "
                "Set it in .streamlit/secrets.toml or as an environment variable."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def chat(
    prompt: str,
    system: str | None = None,
    temperature: float = 0.2,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Send a single-turn chat to OpenAI and return the assistant's reply as text.

    Args:
        prompt: The user's message (what you want answered).
        system: Optional system instruction (tone/role guidance).
        temperature: 0.0–1.0; higher is more creative, lower is more focused.
        model: OpenAI Chat Completions model name.

    Returns:
        str: Assistant response text.
    """
    client = _get_client()

    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()
