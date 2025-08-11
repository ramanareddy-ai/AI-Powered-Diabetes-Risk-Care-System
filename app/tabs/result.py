"""
Results page for the Diabetes Risk & Care System.

This module provides visualizations and summary statistics of the dataset
used to train the machine learning model. It helps users explore the data
underlying the predictions.
"""

from __future__ import annotations

"""
Human-friendly Results page.

Shows a quick snapshot of the latest diagnosis and a basic visualization.
If no diagnosis has been run yet, it guides the user back to the Diagnosis tab.
"""

import streamlit as st
import matplotlib.pyplot as plt

def render() -> None:
    st.header("Results")

    # Expect these to be set by the Diagnosis tab
    prob = st.session_state.get("latest_prob")
    label = st.session_state.get("latest_label")
    inputs = st.session_state.get("latest_inputs")
    explanation = st.session_state.get("latest_explanation")

    if prob is None or label is None or inputs is None:
        st.info("No results yet. Please run a diagnosis first.")
        return

    st.subheader("Summary")
    st.write(f"**Risk Score:** {prob * 100:.1f}%")
    st.write(f"**Category:** {label}")

    st.markdown("**Inputs used**")
    st.json(inputs)

    # Minimal chart (single plot, default colors)
    fig = plt.figure()
    plt.bar(["Risk"], [prob])
    plt.ylim(0, 1)
    st.pyplot(fig)

    if explanation:
        st.markdown("### AI Explanation")
        st.write(explanation)
