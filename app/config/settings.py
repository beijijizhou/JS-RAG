from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    google_gemini_api_key: str = ""  # Set default or load from .env

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()