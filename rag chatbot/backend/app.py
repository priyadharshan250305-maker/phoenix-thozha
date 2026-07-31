from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from embedding import create_embedding
from vector_store import add_document, search
from rag import process_pdf, ask_question
from text_splitter import split_text
app = FastAPI(title="Domain RAG Chatbot")

# React connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.get("/")
def home():
    return {
        "success": True,
        "message": "🚀 Domain RAG Backend Running"
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks = process_pdf(file_path)

        return {
            "success": True,
            "message": "PDF Uploaded Successfully",
            "chunks": chunks
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/chat")
async def chat(data: dict):
    try:
        question = data.get("question")

        answer = ask_question(question)

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }