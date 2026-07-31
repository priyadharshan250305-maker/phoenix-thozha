from math import sqrt

documents = []

def add_document(chunk, embedding):
    documents.append({
        "chunk": chunk,
        "embedding": embedding
    })

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = sqrt(sum(a * a for a in v1))
    mag2 = sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0

    return dot / (mag1 * mag2)

def search(query_embedding, top_k=3):
    scores = []

    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)

    return [doc for score, doc in scores[:top_k]]