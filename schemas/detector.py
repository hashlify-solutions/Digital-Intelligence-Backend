from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from bson import ObjectId


class DetectorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Name of the detector")
    type: Literal["person", "object"] = Field(..., description="Type of detector: person or object")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")


class DetectorResponse(BaseModel):
    id: str = Field(..., alias="_id")
    case_id: str
    name: str
    type: Literal["person", "object"]
    description: Optional[str] = None
    image_path: str
    has_embedding: bool = False
    created_at: datetime
    updated_at: datetime
    user_id: str

    class Config:
        populate_by_name = True


class DetectorUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)


class DetectorMatch(BaseModel):
    id: str = Field(..., alias="_id")
    case_id: str
    detector_id: str
    detector_name: str
    detector_type: Literal["person", "object"]
    detected_item_type: Literal["face", "object"]
    detected_item_id: str
    detected_item_collection: str
    detected_item_path: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: Literal["high", "medium", "low"]
    match_threshold: float = Field(..., ge=0.0, le=1.0)
    created_at: datetime
    
    # Optional metadata from detected item
    source_image_path: Optional[str] = None
    frame_number: Optional[int] = None
    source_video_path: Optional[str] = None

    class Config:
        populate_by_name = True


class DetectorMatchSummary(BaseModel):
    case_id: str
    total_matches: int
    high_confidence_matches: int
    medium_confidence_matches: int
    low_confidence_matches: int
    detector_stats: dict
    matches: List[DetectorMatch]


class DetectorSettings(BaseModel):
    case_id: str
    face_thresholds: dict = Field(default={
        "high_confidence": 0.9,
        "medium_confidence": 0.75,
        "low_confidence": 0.6,
        "minimum_match": 0.96
    })
    object_thresholds: dict = Field(default={
        "high_confidence": 0.85,
        "medium_confidence": 0.7,
        "low_confidence": 0.55,
        "minimum_match": 0.96
    })
    user_id: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        populate_by_name = True


class DetectorSettingsUpdate(BaseModel):
    face_thresholds: Optional[dict] = None
    object_thresholds: Optional[dict] = None


class AnalyzeDetectorsRequest(BaseModel):
    detector_type: Optional[Literal["person", "object"]] = None
    recompute_embeddings: bool = False
    similarity_threshold_override: Optional[float] = Field(None, ge=0.0, le=1.0)


class DetectorAnalysisResponse(BaseModel):
    case_id: str
    analysis_started: bool
    task_id: str
    message: str
    detectors_processed: int
    detected_items_to_analyze: int
