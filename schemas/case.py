from pydantic import BaseModel, Field
from typing import List

class GetMessagesByIdsRequest(BaseModel):
    case_id: str
    message_ids: List[str]
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=10, le=100)
    
class RAGQueryAnalyticsRequest(BaseModel):
    case_id: str
    mongo_ids: List[str] 