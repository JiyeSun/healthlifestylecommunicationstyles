import os
import fitz
from typing import List
import numpy as np
import streamlit as st
from openai import OpenAI

# Initialize the OpenAI client (ensure your API key is set)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper Functions ---

def extract_text_chunks(folder_path="data", chunk_size=300) -> List[str]:
    """Extracts text chunks from PDF files in a folder."""
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                with fitz.open(os.path.join(folder_path, filename)) as doc:
                    for page in doc:
                        text = page.get_text().strip().replace("\n", " ")
                        words = text.split()
                        for i in range(0, len(words), chunk_size):
                            chunk = " ".join(words[i:i + chunk_size])
                            if chunk:
                                chunks.append(chunk)
            except Exception as e:
                st.error(f"Error processing {filename}: {e}")
    return chunks

@st.cache_data  # Cache the embeddings for efficiency
def get_embeddings(texts: List[str]) -> List[List[float]]:  # Specify return type
    """Gets embeddings for a list of texts."""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Or your desired model
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        st.error(f"Error getting embeddings: {e}")
        return []

def get_knowledge_context(query: str, embeddings: List[List[float]], chunks: List[str], top_k: int = 3) -> str:
    """Retrieves the most relevant knowledge chunks for a query."""

    if not embeddings:
        return ""  # Return empty string if no embeddings

    query_embedding = get_embeddings([query])[0]
    scores = np.dot(np.array(embeddings), np.array(query_embedding))
    top_indices = np.argsort(scores)[-top_k:][::-1]
    top_chunks = [chunks[i] for i in top_indices]
    return "\n".join(top_chunks)

def generate_response(query: str, context: str) -> str:
    """Generates a response using the OpenAI API."""
    try:
        prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        response = client.completions.create(  # Or client.chat.completions.create for chat models
            model="gpt-3.5-turbo-instruct",  # Or a suitable model like "gpt-3.5-turbo" for chat
            prompt=prompt,
            max_tokens=250
        )
        return response.choices[0].text.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

# --- Streamlit UI ---

def main():
    st.title("PDF Knowledge Bot")

    # Initialize session state for persistent data
    if 'doc_chunks' not in st.session_state:
        st.session_state.doc_chunks = []
    if 'chunk_embeddings' not in st.session_state:
        st.session_state.chunk_embeddings = []  # Initialize as empty list
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # File uploader
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}"):
                    # Save uploaded file
                    temp_path = os.path.join("./temp", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())

                    new_chunks = extract_text_chunks(folder_path="./temp")
                    st.session_state.doc_chunks.extend(new_chunks)
                    os.remove(temp_path)  # Clean up temp file

                if st.session_state.doc_chunks:
                    with st.spinner("Generating Embeddings"):
                        st.session_state.chunk_embeddings = get_embeddings(st.session_state.doc_chunks)
            except Exception as e:
                st.error(f"Error processing files: {e}")

    # Chat interface
    query = st.text_input("Ask a question:", key="input_query")
    if query:
        if st.session_state.chunk_embeddings:
            context = get_knowledge_context(query, st.session_state.chunk_embeddings, st.session_state.doc_chunks)
            response = generate_response(query, context)

            st.session_state.chat_history.append({"question": query, "answer": response})

            # Display chat history
            for chat in st.session_state.chat_history:
                st.info(f"**You:** {chat['question']}")
                st.success(f"**Bot:** {chat['answer']}")
        else:
            st.warning("Please upload and process PDF files first.")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
