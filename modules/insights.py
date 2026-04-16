# modules/insights.py
import os
from dotenv import load_dotenv
from google import genai


load_dotenv()


def get_client():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not found in .env")
    return genai.Client(api_key=api_key)


# Reuse this client across modules
client = get_client()


def generate_insights(context: str, query: str) -> str:
    prompt = f"""
You are a data analyst.

Context:
{context}

Question:
{query}

Give clear, concise insights in bullet points.
"""
    response = client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt,
    )
    return response.text or ""


