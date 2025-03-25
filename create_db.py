from langchain_community.document_loaders import DirectoryLoader  # Fixed import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
import os
import shutil
from typing import List
# Rest of your code remains unchanged
CHROMA_PATH = "chroma"
DATA_PATH = "javascript"

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", recursive=True)
    documents = loader.load()
    
    for doc in documents:
        relative_path = os.path.relpath(doc.metadata["source"], DATA_PATH)
        print(f"Relative Path: {relative_path}")
        print(f"Content (first 200 chars): {doc.page_content[:200]}...")
        print("-" * 50)
    
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks
class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedder = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")  # Use custom class
    db = Chroma.from_documents(
        chunks,
        embedding=embedder,
        persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()