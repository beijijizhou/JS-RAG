from pydantic import BaseModel

class TextInput(BaseModel):
    texts: list[str]