import streamlit as st
import requests
import json
from pydantic import BaseModel
from typing import List

# streamlit run main.py

# --- Pydantic Models ---
# We can redefine the models here to help with type hinting and clarity
# (This isn't strictly necessary but is good practice)
class SentenceResult(BaseModel):
    sentence: str
    verdict: str
    details: str
    highlight: str


class CheckResponse(BaseModel):
    original_response: str
    results: List[SentenceResult]


# --- App Configuration ---
st.set_page_config(
    page_title="Fact-Checker AI",
    page_icon="🤖",
    layout="wide"
)

# The URL where your FastAPI backend is running
BACKEND_URL = "http://127.0.0.1:8000/check-prompt"

# --- UI Layout ---
st.title("Hallucination & Fact-Checker AI 🕵️")
st.markdown(
    "Enter a prompt to get a response from Gemini, which will then be fact-checked sentence-by-sentence against Wikipedia.")

# Use a session state to keep our results
if 'results' not in st.session_state:
    st.session_state.results = None

# Create a text area for the prompt
user_prompt = st.text_area("Enter your prompt:", height=100,
                           placeholder="e.g., 'What is the fastest animal? Describe it.'")

# Create a submit button
if st.button("Check Prompt", type="primary", use_container_width=True):
    if user_prompt:
        st.session_state.results = None  # Clear old results

        # Show a loading spinner while we wait for the backend
        with st.spinner("Calling Gemini, running models, and checking facts... this might take a moment."):
            try:
                # 1. Prepare the request
                payload = {"prompt": user_prompt}

                # 2. Call the backend API
                response = requests.post(BACKEND_URL, json=payload)

                # 3. Handle the response
                if response.status_code == 200:
                    # Success! Store the results
                    st.session_state.results = response.json()
                else:
                    # Show an API error
                    st.error(f"Error from backend: {response.json().get('detail', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error(f"❌ Connection Error: Could not connect to the backend at {BACKEND_URL}. Is it running?")
            except Exception as e:
                st.error(f"An unknown error occurred: {e}")
    else:
        st.warning("Please enter a prompt.")

# --- Display Results ---
if st.session_state.results:

    st.divider()
    st.subheader("Analysis Results")

    # Parse the data with our Pydantic model
    try:
        response_data = CheckResponse.parse_obj(st.session_state.results)

        # Display each sentence with its highlight
        for result in response_data.results:

            if result.highlight == "green":
                st.success(result.sentence)
            elif result.highlight == "red":
                st.error(result.sentence)
            else:  # 'none'
                st.write(result.sentence)

            # Show details in an expander
            with st.expander("Show Details"):
                st.markdown(f"**Verdict:** {result.verdict}")
                st.markdown(f"**Details:** {result.details}")

    except Exception as e:
        st.error(f"Error parsing backend response: {e}")
        st.json(st.session_state.results)  # Show the raw JSON if parsing fails