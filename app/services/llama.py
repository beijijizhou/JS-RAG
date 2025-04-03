import os
from pinecone import Pinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import numpy as np

# Pinecone setup
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("API key not set. Please set PINECONE_API_KEY.")
pc = Pinecone(api_key=api_key)
index_name = "testing"
index = pc.Index(index_name)
