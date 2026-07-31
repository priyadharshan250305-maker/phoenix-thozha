import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")


def ask_gemini(context, question):
    prompt = f"""
You are a Domain RAG Chatbot.

Rules:
1. Answer ONLY from the provided context.
2. If the answer is not in the context, reply exactly:
I don't know.

Context:
{context}

Question:
{question}
"""

    response = model.generate_content(prompt)

    return response.text