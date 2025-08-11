"""
Home page for the Diabetes Risk & Care System.

This module defines the ``render()`` function that displays an overview
of the application and guides the user to the other sections.
"""

import streamlit as st


def render() -> None:
    """Render the home page."""
    st.title("AI‑Powered Diabetes Risk & Care System")
    st.markdown(
        """
        Welcome to the **AI‑Powered Diabetes Risk & Care System**. This application
        leverages both **machine learning** and **artificial intelligence** to help
        you understand your potential risk of developing diabetes.

        - In the **Diagnosis** tab you can enter your personal health metrics and
          receive a probability estimate from a logistic regression model trained on
          real patient data.
        - The prediction is accompanied by a **human‑friendly explanation** generated
          by a language model to help you interpret the result.
        - The **Results** tab provides an overview of the dataset and visualizations
          to help you explore the underlying data.
        - The **Knowledge Center** offers general information about diabetes and
          prevention strategies.
        - The **Talk to Doctor** tab allows you to ask questions and receive
          AI‑generated answers from a virtual physician.

        **Disclaimer:** This tool is for educational purposes only and does not
        constitute medical advice. Always consult a healthcare professional for
        medical concerns.
        """
    )
