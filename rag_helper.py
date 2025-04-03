
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.readers.web import SimpleWebPageReader

def build_knowledge_index(pdf_folder="data", urls=None, save_path="my_knowledge_index"):
    if urls is None:
        urls = []

    pdf_docs = SimpleDirectoryReader(pdf_folder).load_data()
    web_docs = []
    if urls:
        web_loader = SimpleWebPageReader()
        web_docs = web_loader.load_data(urls)

    docs = pdf_docs + web_docs
    index = GPTVectorStoreIndex.from_documents(docs)
    index.storage_context.persist(save_path)
    print(f"Index built and saved to: {save_path}")

def get_knowledge_context(query, index_path="my_knowledge_index", top_k=3):
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_docs = retriever.retrieve(query)
    context = "\n".join([doc.text for doc in retrieved_docs])
    return context

