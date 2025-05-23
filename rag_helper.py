import os
import fitz
from typing import List
import numpy as np
import streamlit as st
from openai import OpenAI
#client = OpenAI()
# Initialize the OpenAI client (ensure your API key is set)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#openai.api_key = os.getenv("OPENAI_API_KEY")
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DOC_CHUNKS = []
CHUNK_EMBEDDINGS = []


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


def get_embeddings(texts: List[str]) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([item.embedding for item in response.data])
    #return np.array([r["embedding"] for r in response["data"]])

def init_knowledge_base(folder_path="data"):
    global DOC_CHUNKS, CHUNK_EMBEDDINGS
    DOC_CHUNKS = extract_text_chunks(folder_path)
    CHUNK_EMBEDDINGS = get_embeddings(DOC_CHUNKS)


def get_knowledge_context(query: str, top_k: int = 3) -> str:
    global DOC_CHUNKS, CHUNK_EMBEDDINGS
    if CHUNK_EMBEDDINGS is None or len(CHUNK_EMBEDDINGS) == 0:
        raise ValueError("Knowledge base is not initialized.")

    query_embedding = get_embeddings([query])[0]
    scores = np.dot(CHUNK_EMBEDDINGS, query_embedding)
    top_indices = scores.argsort()[-top_k:][::-1]
    top_chunks = [DOC_CHUNKS[i] for i in top_indices]
    return "\n".join(top_chunks)
