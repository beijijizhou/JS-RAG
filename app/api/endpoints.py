from fastapi import APIRouter
from app.models.text_input import TextInput
from app.services.embedding import get_embedding
from app.services.chroma import query_chroma

router = APIRouter()

@router.post("/embed")
async def embed(input: TextInput):
    """Embed the input text and query ChromaDB for relevant documents."""
    query = input.texts[0]  # Take first text
    query_embedding = get_embedding(query)
    results = query_chroma(query_embedding)
    return {
        "query": query,
        "results": results
    }