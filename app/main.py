"""
Streamlit entrypoint for the Diabetes Risk & Care System.

This module creates a sidebar navigation and delegates rendering
to individual page modules. Pages are registered in the ``app.tabs``
package.

To run this application locally, execute:

.. code-block:: bash

    streamlit run app/main.py

from the project root.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.tabs import home, diagnosis, result, kc, talk2doc


def main() -> None:
    """Render the application sidebar and dispatch to selected page.

    This function builds a simple sidebar allowing the user to switch
    between the different areas of the app. Each page is implemented
    in its own module with a ``render()`` function.
    """
    st.set_page_config(page_title="AI‑Powered Diabetes Risk & Care System", page_icon="🩺")
    st.sidebar.title("Navigation")
    pages = {
        "Home": home.render,
        "Diagnosis": diagnosis.render,
        "Results": result.render,
        "Knowledge Center": kc.render,
        "Talk to Doctor": talk2doc.render,
    }
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    # Call the appropriate page's render function
    pages[choice]()


if __name__ == "__main__":
    # When run as a script, launch the app
    main()