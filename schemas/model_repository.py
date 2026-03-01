from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ModelRepository(BaseModel):
    model: str
    name: str
    type: str
    is_local: bool
    base_type: str
    embedding_size: Optional[int] = None
    created_at: Optional[str] = None 