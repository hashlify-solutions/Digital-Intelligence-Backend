from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class UfdrFileCreate(BaseModel):
    name: str
    caseId: str
    file_size: int


class UfdrFileInDB(BaseModel):
    _id: str
    name: str
    caseId: str
    file_size: int
    associated_schema_names: List[str] = []
    created_at: datetime
    updated_at: datetime
