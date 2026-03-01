from pydantic import BaseModel
from typing import List, Optional, Dict


class ProcessingProfile(BaseModel):
    name: str
    description: Optional[str] = None
    classifier: Dict[str, Optional[str | bool]]
    emotion: Dict[str, Optional[str | bool]]
    embeddings: Dict[str, Optional[str | bool | int]]
    toxic: Dict[str, Optional[str | bool]]
    
    # user_id: Optional[str] = Field(None, alias="_id")  # MongoDB User ID reference

