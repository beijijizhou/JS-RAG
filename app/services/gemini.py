import google.generativeai as genai
from app.config.settings import settings
import re
from typing import AsyncGenerator
# Configure the API key once at module load
genai.configure(api_key=settings.google_gemini_api_key)

# Initialize the Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")
print("Loaded AI")
# Base prompt
BASE_PROMPT = (
    "You are a JavaScript expert. Based on the provided context, provide a clear and concise explanation "
    "of the query in the context of JavaScript programming. If applicable, include a JavaScript code example "
    "in a Markdown code block (```javascript). Separate the explanation from the code clearly.\n\n"
)

async def generate_response(query: str, context: list[str]) -> tuple[str, str | None]:
    """
    Generate a response using Gemini based on the query and context.
    Returns a tuple of (explanation, code).
    """
    # Construct the prompt
    prompt = (
        BASE_PROMPT +
        f"Query: {query}\n\n" +
        "Context:\n" +
        "\n".join([f"- {doc}" for doc in context])
    )

    try:
        # Generate content (async call)
        response = await model.generate_content_async(prompt)
        response_text = response.text

        # Extract code from Markdown code block
        code_match = re.search(r"```(?:javascript|js)?\n([\s\S]*?)\n```", response_text)
        code = code_match.group(1).strip() if code_match else None

        # Remove code block from explanation
        explanation = (
            re.sub(r"```(?:javascript|js)?\n[\s\S]*?\n```", "", response_text)
            .strip()
        )

        return explanation, code
    except Exception as e:
        return f"Error generating response: {str(e)}", None
    
async def generate_response_stream(query: str, context: list[str]) -> AsyncGenerator[str, None]:
    """Streaming response yielding chunks of text."""
    prompt = BASE_PROMPT + f"Query: {query}\n\nContext:\n" + "\n".join([f"- {doc}" for doc in context])
    try:
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            yield chunk.text
    except Exception as e:
        yield f"Error generating response: {str(e)}"