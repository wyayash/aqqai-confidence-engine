import os
import chromadb
import hashlib

def get_embedding(text):
    
    vector = [0.0] * 384
    words = text.lower().replace('.', '').replace(',', '').split()
    for word in words:
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 384
        vector[idx] += 1.0
        
    norm = sum(x**2 for x in vector) ** 0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    return vector

def ingest_documents():
    docs = [
        "AQQAI is building a next-generation AI chip that replaces NVIDIA's H100.",
        "The core innovation is using magnetite (Fe3O4) from Indian beach sand as a spintronic memory layer combined with silicon logic.",
        "AQQAI chips result in 10x faster memory reads, 50W vs 700W power draw, and a fraction of the manufacturing cost.",
        "The one unsolved problem is reliable magnetic domain alignment of magnetite at chip scale."
    ]
    
    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        client.delete_collection("aqqai_docs")
    except Exception:
        pass
        
    collection = client.create_collection(name="aqqai_docs")
    
    print("Embedding and storing documents via Zero-Dependency Local Engine...")
    for i, doc in enumerate(docs):
        emb = get_embedding(doc)
        collection.add(
            documents=[doc],
            embeddings=[emb],
            ids=[f"doc_{i}"]
        )
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_documents()