from pydantic import BaseModel
from typing import List, Optional


class AlertModel(BaseModel):
    toxicity_score: Optional[float] = None
    risk_level: Optional[List[str]] = None
    sentiment_aspect: Optional[List[str]] = None
    emotion: Optional[List[str]] = None
    language: Optional[List[str]] = None
    interaction_type: Optional[List[str]] = None
    top_topic: Optional[List[str]] = None
    description: Optional[str] = None
    entities: Optional[List[str]] = None
    name: str
    # user_id: Optional[str] = Field(None, alias="_id")  # MongoDB User ID reference

