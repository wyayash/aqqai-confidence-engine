import os
import chromadb
import hashlib
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

def get_embedding(text):
    vector = [0.0] * 384
    words = text.lower().replace('.', '').replace(',', '').replace('?', '').split()
    for word in words:
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % 384
        vector[idx] += 1.0
        
    norm = sum(x**2 for x in vector) ** 0.5
    if norm > 0:
        vector = [x / norm for x in vector]
    return vector

def query_rag(user_query):
    print(f"\nQuerying: '{user_query}'")
    
    query_emb = get_embedding(user_query)
        
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="aqqai_docs")
    
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=2
    )
    
    retrieved_chunks = results['documents'][0]
    context = "\n".join(retrieved_chunks)
    print(f"Retrieved Context:\n{context}\n")
    
    prompt = f"""System: You are a precise assistant. Answer ONLY using the provided context. If the answer is not in the context, say "I don't know."
    
    Context:
    {context}
    
    Question: {user_query}
    Answer:"""
    
    print("Generating response via Groq (llama-3.3-70b-versatile)...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    
    answer = chat_completion.choices[0].message.content
    print(f"\nFinal Answer:\n{answer}")
    return answer