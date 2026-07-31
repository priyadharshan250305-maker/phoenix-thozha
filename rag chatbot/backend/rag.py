from pypdf import PdfReader
from gemini_service import ask_gemini
from text_splitter import split_text
from embedding import create_embedding
from vector_store import add_document, search
document_chunks = []

document_chunks = []

def process_pdf(file_path):
    global document_chunks

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    # Split text into chunks
    document_chunks = split_text(text)

    # Clear old documents
    from vector_store import documents
    documents.clear()

    # Create embeddings and store them
    for chunk in document_chunks:
        embedding = create_embedding(chunk)
        add_document(chunk, embedding)

    return len(document_chunks)

    document_chunks = [
        text[i:i+chunk_size]
        for i in range(0, len(text), chunk_size)
    ]

    return len(document_chunks)


def ask_question(question):

    query_embedding = create_embedding(question)

    results = search(query_embedding)

    context = "\n\n".join(
        [doc["chunk"] for doc in results]
    )

    if not context:
        return "I don't know."

    answer = ask_gemini(context, question)

    return answer