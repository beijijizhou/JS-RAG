from sentence_transformers import SentenceTransformer

# Load model once at startup
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text string."""
    return embedder.encode(text, convert_to_tensor=False).tolist()