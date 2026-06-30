import os
from ingest import ingest_documents
from query import query_rag

def main():
    print("="*60)
    print(" AQQAI — RAG PROTOTYPE v5.0 (FIREWALL BYPASS)")
    print(" Embeddings: Zero-Dependency Local Hash Engine")
    print(" LLM: Llama 3.3 via Groq API")
    print("="*60)
    
    if not os.environ.get("GROQ_API_KEY"):
        print("\nERROR: Missing API Key. Ensure GROQ_API_KEY is set.")
        return

    print("\n--- PHASE 1: INGESTION ---")
    ingest_documents()
    
    print("\n--- PHASE 2: QUERYING ---")
    query_rag("What is the core innovation of AQQAI and what is the one unsolved problem?")

if __name__ == "__main__":
    main()