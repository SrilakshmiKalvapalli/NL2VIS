# config/gemini_config.py
import os
from dotenv import load_dotenv
from google import genai

def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
    client = genai.Client(api_key=api_key)
    return client


