import fitz  # PyMuPDF
import os

def get_pdf_text(folder_path="data"):
    text_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            with fitz.open(full_path) as doc:
                for page in doc:
                    text_chunks.append(page.get_text())
    return "\n".join(text_chunks)

def get_context(query=None, folder_path="data", max_chars=3000):
    full_text = get_pdf_text(folder_path)
    # Optional: Simple truncation to stay within token limits
    if len(full_text) > max_chars:
        return full_text[:max_chars]
    return full_text
