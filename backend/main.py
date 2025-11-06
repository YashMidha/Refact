import requests
from bs4 import BeautifulSoup
import torch
import spacy
# import pandas as pd
import re
# import fastcoref  # <-- CHANGED
from fastcoref import spacy_component
import os  # <-- For API key
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google import genai
from google.genai.errors import APIError
from typing import List, Dict, Any

# --- FastAPI & Pydantic Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# ----------------- API KEY -----------------------
# set GOOGLE_API_KEY="your_actual_key_here"
# uvicorn main:app
# uvicorn main:app --reload

# --- 1. DEFINE OUR DATA MODELS ---
# Pydantic models define the "shape" of our API data

class PromptRequest(BaseModel):
    """The data we expect from the frontend."""
    prompt: str


class SentenceResult(BaseModel):
    """The data we'll send back for *each* sentence."""
    sentence: str
    verdict: str
    details: str
    highlight: str  # <-- We'll pre-calculate the color here: 'red', 'green', 'none'


class CheckResponse(BaseModel):
    """The final, complete response for the frontend."""
    original_response: str
    results: List[SentenceResult]


# --- 2. MODEL CACHE ---
# This dict will hold our heavy AI models so we only load them once.
model_cache: Dict[str, Any] = {}


# --- 3. LIFESPAN FUNCTION (THE SMART WAY TO LOAD MODELS) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """This function runs ONCE when the server starts."""
    print("Server starting up...")

    # --- Load NLI Model ---
    print("Loading NLI model...")
    model_cache["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model_cache["nli_tokenizer"] = AutoTokenizer.from_pretrained("my_specialized_model")
        model_cache["nli_model"] = AutoModelForSequenceClassification.from_pretrained("my_specialized_model").to(
            model_cache["device"])
        model_cache["nli_ready"] = True
    except Exception as e:
        print(f"!!! FATAL: Could not load NLI model. {e}")
        model_cache["nli_ready"] = False

    # --- Load spaCy ---
    print("Loading spaCy model (en_core_web_trf) and fastcoref...")  # <-- CHANGED
    try:
        model_cache["nlp"] = spacy.load("en_core_web_trf")
        model_cache["nlp"].add_pipe('fastcoref')  # <-- CHANGED
    except Exception as e:
        print(f"!!! FATAL: Could not load spaCy. {e}")
        # Handle error appropriately, maybe exit

    # --- Load Gemini Client ---
    print("Initializing Gemini client...")
    try:
        # NOTE: Set GOOGLE_API_KEY in your environment!
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("!!! WARNING: GOOGLE_API_KEY not set. Gemini calls will fail.")

        model_cache["gemini_client"] = genai.Client(api_key=api_key)
    except Exception as e:
        print(f"!!! FATAL: Could not initialize Gemini client. {e}")

    print("--- Models loaded. Server is ready. ---")

    yield  # <-- This is where the application runs

    # --- Code after 'yield' runs on shutdown ---
    print("Server shutting down...")
    model_cache.clear()


# --- 4. CREATE THE FASTAPI APP ---
# We pass in our 'lifespan' function to load models on startup
app = FastAPI(lifespan=lifespan)


# --- 5. YOUR PIPELINE LOGIC (Refactored) ---
# All your functions go here, slightly modified to use the model_cache

# --- Gemini API Call ---
def get_gemini_answer(prompt: str) -> str:
    client = model_cache.get("gemini_client")
    if not client:
        raise HTTPException(status_code=500, detail="Gemini client not initialized.")
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0
            )
        )
        return response.text.strip()
    except APIError as e:
        print(f"Gemini API Error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


# --- Text Cleaning ---
def clean_gemini_response(text: str) -> str:
    text = text.replace('*', '')
    text = re.sub(r'[\n\r]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# --- Wiki Context Scraper ---
def get_cleaned_wiki_text(page_title: str):
    URL = f"https://en.wikipedia.org/w/api.php"
    HEADERS = {'User-Agent': 'MyFactCheckerBot/1.0 (contact@example.com)'}
    PARAMS = {"action": "parse", "page": page_title, "prop": "text", "format": "json", "redirects": ""}
    try:
        response = requests.get(url=URL, params=PARAMS, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        html_content = data.get("parse", {}).get("text", {}).get("*", "")
        if not html_content: return None
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup.find_all(['sup', 'table']): element.decompose()
        return '\n'.join([line.strip() for line in soup.get_text().split('\n') if line.strip()])[:5000]
    except requests.exceptions.RequestException:
        return None


# --- Topic Finding Logic ---
BAD_ENTITY_LABELS = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"}
GENERIC_SUBJECTS = ['people', 'it', 'they', 'he', 'she', 'things', 'many', 'some', 'i', 'you', 'we', 'that', 'this']


def get_topic_priority_list(doc) -> list[str]:
    entities = [e.text.strip(".,;") for e in doc.ents if e.label_ not in BAD_ENTITY_LABELS]
    if entities:
        return sorted(entities, key=len, reverse=True)[:3]
    noun_chunks = [chunk.text.strip(".,;") for chunk in doc.noun_chunks]
    good_noun_chunks = [c for c in sorted(noun_chunks, key=len, reverse=True) if c.lower() not in GENERIC_SUBJECTS]
    if good_noun_chunks:
        return good_noun_chunks[:3]
    return []


# --- SINGLE SENTENCE CHECKER (Refactored) ---
# This is your key function, refactored to use the cached models
def check_sentence(sentence: str) -> (str, str, str):
    """
    Checks one sentence.
    Returns: (verdict_string, details_string, highlight_color)
    """
    if not model_cache.get("nli_ready"):
        return "Uncertain", "NLI Model Not Loaded", "none"

    # Get models from cache
    nlp = model_cache["nlp"]
    tokenizer = model_cache["nli_tokenizer"]
    model = model_cache["nli_model"]
    device = model_cache["device"]
    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

    doc = nlp(sentence)
    topic_list = get_topic_priority_list(doc)

    if not topic_list:
        return "Uncertain", "Could not find any specific topics.", "none"

    context = None
    final_topic = None
    for topic in topic_list:
        context = get_cleaned_wiki_text(topic)
        if context:
            final_topic = topic
            break

    if not context:
        return "Uncertain", f"Could not find context for topics: {topic_list}", "none"

    inputs = tokenizer(context, sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][pred_id].item()

    verdict = label_map.get(pred_id, "Unknown")
    details = f"Topic: '{final_topic}' | Context from Wiki: '{context[:100]}...'"

    # --- This is the new logic for the frontend ---
    if verdict == "contradiction":
        hallucination_verdict = f"Hallucination (Contradiction: {confidence:.2%})"
        highlight = "red"
    elif verdict == "neutral":
        hallucination_verdict = f"Possible Hallucination (Neutral: {confidence:.2%})"
        highlight = "none"
    else:
        hallucination_verdict = f"Supported (Entailment: {confidence:.2%})"
        highlight = "green"
    # --- End new logic ---

    return hallucination_verdict, details, highlight


# --- 6. THE API ENDPOINT ---
# This is the URL the frontend will call
@app.post("/check-prompt")
async def check_prompt(request: PromptRequest) -> CheckResponse:
    """
    The main API endpoint. Receives a prompt, runs the full pipeline,
    and returns the structured sentence-by-sentence results.
    """
    print(f"--- Received prompt: '{request.prompt}' ---")

    # 1. Get answer from Gemini
    gemini_answer_raw = get_gemini_answer(request.prompt)
    print(f"Raw Gemini Answer: {gemini_answer_raw}")

    # 2. Clean and Resolve Pronouns
    cleaned_answer = clean_gemini_response(gemini_answer_raw)
    # Pass component_cfg to trigger text resolution as per your logic
    doc_with_corefs = model_cache["nlp"](cleaned_answer, component_cfg={"fastcoref": {'resolve_text': True}})
    resolved_answer = doc_with_corefs._.resolved_text  # <-- CHANGED attribute
    print(f"Resolved Answer: {resolved_answer}")

    # 3. Split into sentences
    doc_resolved = model_cache["nlp"](resolved_answer)
    sentences = [sent.text.strip() for sent in doc_resolved.sents if sent.text.strip()]

    if not sentences:
        return CheckResponse(
            original_response=resolved_answer,
            results=[]
        )

    # 4. Check each sentence
    print(f"Checking {len(sentences)} sentences...")
    final_results: List[SentenceResult] = []
    for sentence in sentences:
        verdict, details, highlight = check_sentence(sentence)
        final_results.append(
            SentenceResult(
                sentence=sentence,
                verdict=verdict,
                details=details,
                highlight=highlight
            )
        )
        print(f"-> '{sentence}'")
        print(f"   => {verdict}")

    # 5. Return the full, structured response
    print("--- Check complete. Sending response. ---")
    return CheckResponse(
        original_response=resolved_answer,
        results=final_results
    )


# --- 7. (Optional) A simple root endpoint ---
@app.get("/")
def read_root():
    return {"Hello": "This is your Fact-Checking API. POST to /check-prompt."}