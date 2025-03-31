from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from app.models.text_input import TextInput
from app.services.embedding import get_embedding
from app.services.chroma import query_chroma
from app.services.gemini import generate_response, generate_response_stream

router = APIRouter()

@router.post("/embed")
async def embed(input: TextInput):
    """Non-streaming endpoint."""
    query = input.texts[0]
    query_embedding = get_embedding(query)
    results = query_chroma(query_embedding)

    cleaned_results = [
        result["document"].split("\n\n", 1)[-1].strip()
        for result in results
    ]

    explanation, code = await generate_response(query, cleaned_results)

    return {
        "query": query,
        "results": results,
        "explanation": explanation,
        "code": code
    }

@router.post("/embed/stream")
async def embed_stream(input: TextInput):
    """Streaming endpoint."""
    query = input.texts[0]
    query_embedding = get_embedding(query)
    results = query_chroma(query_embedding)

    cleaned_results = [
        result["document"].split("\n\n", 1)[-1].strip()
        for result in results
    ]

    async def stream_response():
        async for chunk in generate_response_stream(query, cleaned_results):
            yield chunk

    return StreamingResponse(stream_response(), media_type="text/plain")