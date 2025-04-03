from asyncio import sleep
import os
import time
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Pinecone setup
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("API key not set. Please set PINECONE_API_KEY.")
pc = Pinecone(api_key=api_key)
index_name = "testing"
index = pc.Index(index_name)
print("index connected")
# Paths
DATA_PATH = "javascript"
NAMESPACE = "js-rag"

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="**/*.md", recursive=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
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

def prepare_records(chunks: list[Document]):
    records = []
    print("prepare_records")

    for i, chunk in enumerate(chunks):
        file_path = chunk.metadata.get("source", "unknown")
        relative_path = os.path.relpath(file_path, DATA_PATH)
        chunk_id = f"{relative_path.replace(os.sep, '_')}_chunk_{i}"
        record = {
            "id": chunk_id,
            "text": chunk.page_content
        }
        records.append(record)
    return records

def upsert_to_pinecone(records: list[dict]):
    # Batch upsert (Pinecone recommends batches of ~100)
    print("start to upsert")
    batch_size = 32
    for i in range(0, len(records), batch_size):
        batch_records = records[i:i + batch_size]
        index.upsert_records(
            namespace=NAMESPACE,
            records=batch_records
        )
        time.sleep(3)
        print(f"Uploaded batch {i // batch_size + 1} of {len(records) // batch_size + 1}")
    print(f"Uploaded {len(records)} records to Pinecone index {index_name} in namespace {NAMESPACE}.")

def main():
    # Load and split documents
    documents = load_documents()
    chunks = split_text(documents)
    
    # Prepare records for Pinecone
    records = prepare_records(chunks)
    
    # Upsert to Pinecone
    upsert_to_pinecone(records)
    
    # Verify
    stats = index.describe_index_stats()
    print(f"Index stats after upload: {stats}")

if __name__ == "__main__":
    main()