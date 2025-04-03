import os
import fitz  # PyMuPDF
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load embedding model (you can change to 'all-MiniLM-L6-v2' or OpenAI)
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Global FAISS index and document chunks
INDEX = None
DOC_CHUNKS = []
EMBED_DIM = 384  # Depends on model (MiniLM is 384)


def extract_text_chunks(folder_path="data", chunk_size=300) -> List[str]:
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(folder_path, filename)) as doc:
                for page in doc:
                    text = page.get_text().strip().replace("\n", " ")
                    words = text.split()
                    for i in range(0, len(words), chunk_size):
                        chunk = " ".join(words[i:i + chunk_size])
                        if chunk:
                            chunks.append(chunk)
    return chunks


def build_faiss_index(chunks: List[str]):
    global INDEX, DOC_CHUNKS
    DOC_CHUNKS = chunks
    embeddings = EMBED_MODEL.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.array(embeddings))
    INDEX = index


def get_knowledge_context(query: str, top_k: int = 3) -> str:
    if INDEX is None:
        raise ValueError("FAISS index has not been initialized. Call `init_knowledge_base()` first.")
    
    query_embedding = EMBED_MODEL.encode([query], convert_to_numpy=True)
    D, I = INDEX.search(np.array(query_embedding), top_k)
    top_chunks = [DOC_CHUNKS[i] for i in I[0] if i < len(DOC_CHUNKS)]
    return "\n".join(top_chunks)


def init_knowledge_base(folder_path="data"):
    chunks = extract_text_chunks(folder_path)
    build_faiss_index(chunks)
