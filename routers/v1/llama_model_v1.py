# Standard 3rd party dependencies imports 
from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field
from bson import ObjectId
import datetime
from fastapi.responses import JSONResponse
# Internal local modules imports
from clients.llama.llama_v1 import LlamaClient
from config.db import models_master_collection, db, processing_profiles_collection
from utils.auth import get_current_user
from rag_v1 import ArabicRagAnalyzer

# Router setup
router = APIRouter()

# Create a new collection for tracking profile usage
profile_usage_collection = db["model_profile_usage"]

class LlamaModelSettings(BaseModel):
    basic_params: Optional[Dict[str, Any]] = None
    advanced_params: Optional[Dict[str, Any]] = None
# Type for parameter values matching TypeScript ParamValueType
ParamValueType = Union[str, int, float, bool]

class ParameterMetadata(BaseModel):
    value: ParamValueType
    type: str = Field(..., description="Type of parameter: number, string, boolean")
    description: str
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[ParamValueType]] = None

class ModelParametersMetadata(BaseModel):
    basic_params: Dict[str, ParameterMetadata]
    advanced_params: Dict[str, ParameterMetadata]

class LlamaModelProfileBase(BaseModel):
    name: str
    description: Optional[str] = None
    classifier: Dict[str, Any]
    embeddings: Dict[str, Any]
    emotion: Dict[str, Any]
    toxic: Dict[str, Any]
    llama: Dict[str, Any]  # Should contain 'basic_params', 'advanced_params', and optionally 'prompt_engineering'

class SaveModelProfileResponse(BaseModel):
    id: str
    status: str

class ChatRequest(BaseModel):
    prompt: str
    variables: Dict[str, Any]
    model_profile_id: Optional[str] = None
    basic_params: Optional[Dict[str, Any]] = None
    advanced_params: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    status: str
    usage_id: Optional[str] = None

class ProfileUsageHistory(BaseModel):
    profile_id: str
    profile_name: str
    usage_count: int
    last_used: str
    success_rate: float

def validate_parameters(params: Dict[str, Any], metadata: Dict[str, ParameterMetadata]) -> bool:
    """Validate if the parameters are within allowed ranges and options"""
    if not params:
        return True

    for param_name, param_value in params.items():
        if param_name not in metadata:
            return False

        meta = metadata[param_name]
        
        # Check type
        if meta.type == "number" and not isinstance(param_value, (int, float)):
            return False
        elif meta.type == "string" and not isinstance(param_value, str):
            return False
        elif meta.type == "boolean" and not isinstance(param_value, bool):
            return False

        # Check constraints
        if meta.type == "number":
            if meta.min is not None and param_value < meta.min:
                return False
            if meta.max is not None and param_value > meta.max:
                return False
            if meta.options is not None and param_value not in meta.options:
                return False
        elif meta.type == "string" and meta.options is not None:
            if param_value not in meta.options:
                return False

    return True

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user_id: str = Depends(get_current_user)):
    try:
        # Get model profile if provided
        model_profile = None
        usage_id = None
        
        if request.model_profile_id:
            model_profile = await models_master_collection.find_one({"_id": ObjectId(request.model_profile_id)})
            if not model_profile:
                raise HTTPException(status_code=404, detail="Model profile not found")
            
            # Record profile usage
            usage_record = {
                "profile_id": ObjectId(request.model_profile_id),
                "user_id": user_id,
                "prompt": request.prompt,
                "variables": request.variables,
                "timestamp": datetime.datetime.now(datetime.timezone.utc),
                "status": "started"
            }
            usage_result = await profile_usage_collection.insert_one(usage_record)
            usage_id = str(usage_result.inserted_id)

        # Get parameter metadata for validation
        metadata = get_parameters_metadata()
        
        # Get defaults
        default_basic = {
            "temperature": 0.6,
            "num_ctx": 2048,
            "num_tokens": 512
        }
        default_advanced = {
            "optimizer": "AdamW",
            "batch_size": 32,
            "epochs": 5,
            "learning_rate": 3e-5,
            "dropout_rate": 0.1,
            "early_stopping": True
        }
        # Get from profile if present
        profile_basic = model_profile.get("basic_params") if model_profile else {}
        profile_advanced = model_profile.get("advanced_params") if model_profile else {}
        # Merge: default < profile < user
        basic_params = {**default_basic, **(profile_basic or {}), **(request.basic_params or {})}
        advanced_params = {**default_advanced, **(profile_advanced or {}), **(request.advanced_params or {})}
        
        # Validate parameters if they exist
        if basic_params and not validate_parameters(basic_params, metadata.basic_params):
            raise HTTPException(status_code=400, detail="Invalid basic parameters")
            
        if advanced_params and not validate_parameters(advanced_params, metadata.advanced_params):
            raise HTTPException(status_code=400, detail="Invalid advanced parameters")

        # Initialize LlamaClient with merged params
        client = LlamaClient(
            prompt=request.prompt,
            variables=request.variables,
            basic_params=basic_params,
            advanced_params=advanced_params
        )
        
        response = client.chat()
        
        # Update usage record if it exists
        if usage_id:
            await profile_usage_collection.update_one(
                {"_id": ObjectId(usage_id)},
                {
                    "$set": {
                        "status": "completed",
                        "response": response,
                        "completed_at": datetime.datetime.now(datetime.timezone.utc)
                    }
                }
            )
        
        return ChatResponse(
            response=response,
            status="success",
            usage_id=usage_id
        )
    except Exception as e:
        # Update usage record with error if it exists
        if usage_id:
            await profile_usage_collection.update_one(
                {"_id": ObjectId(usage_id)},
                {
                    "$set": {
                        "status": "failed",
                        "error": str(e),
                        "completed_at": datetime.datetime.now(datetime.timezone.utc)
                    }
                }
            )
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

def get_parameters_metadata() -> ModelParametersMetadata:
    """Generate metadata for all model parameters"""
    basic_params = {
        "temperature": ParameterMetadata(
            value=0.6,
            type="number",
            description="Controls randomness in the output. Higher values make the output more random, lower values make it more deterministic.",
            min=0.0,
            max=2.0
        ),
        "num_ctx": ParameterMetadata(
            value=2048,
            type="number",
            description="The context window size",
            min=512,
            max=4096
        ),
        "num_tokens": ParameterMetadata(
            value=512,
            type="number",
            description="Maximum number of tokens to generate",
            min=64,
            max=2048
        )
    }

    advanced_params = {
        "optimizer": ParameterMetadata(
            value="AdamW",
            type="string",
            description="Optimization algorithm to use",
            options=["AdamW", "Adam", "SGD"]
        ),
        "batch_size": ParameterMetadata(
            value=32,
            type="number",
            description="Training batch size",
            options=[8, 16, 32, 64, 128]
        ),
        "epochs": ParameterMetadata(
            value=5,
            type="number",
            description="Number of training epochs",
            min=1,
            max=100
        ),
        "learning_rate": ParameterMetadata(
            value=3e-5,
            type="number",
            description="Learning rate for optimization",
            min=1e-6,
            max=1e-3
        ),
        "dropout_rate": ParameterMetadata(
            value=0.1,
            type="number",
            description="Dropout rate for regularization",
            min=0.0,
            max=0.5
        ),
        "early_stopping": ParameterMetadata(
            value=True,
            type="boolean",
            description="Whether to use early stopping during training"
        )
    }

    return ModelParametersMetadata(
        basic_params=basic_params,
        advanced_params=advanced_params
    )

@router.post("/save-model-profile", response_model=SaveModelProfileResponse)
async def save_model_profile(
    profile: LlamaModelProfileBase,
    user_id: str = Depends(get_current_user)
):
    """Save a new model profile with all settings (models and llama settings) in MongoDB."""
    try:
        # Validate Llama parameters against metadata
        metadata = get_parameters_metadata()
        llama = profile.llama or {}
        basic_params = llama.get("basic_params", {})
        advanced_params = llama.get("advanced_params", {})
        if basic_params and not validate_parameters(basic_params, metadata.basic_params):
            raise HTTPException(status_code=400, detail="Invalid basic parameters")
        if advanced_params and not validate_parameters(advanced_params, metadata.advanced_params):
            raise HTTPException(status_code=400, detail="Invalid advanced parameters")

        # Store the full model profile in MongoDB
        doc = {
            "name": profile.name,
            "description": profile.description,
            "classifier": profile.classifier,
            "embeddings": profile.embeddings,
            "emotion": profile.emotion,
            "toxic": profile.toxic,
            "llama": profile.llama,  # includes basic, advanced, and prompt_engineering
            "user_id": user_id,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        result = await models_master_collection.insert_one(doc)
        return SaveModelProfileResponse(id=str(result.inserted_id), status="success")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving model profile: {str(e)}"
        )

@router.get("/model-settings", response_model=ModelParametersMetadata)
async def get_model_settings():
    """Get the default model settings with metadata"""
    return get_parameters_metadata()

@router.get("/model-profiles")
async def get_model_profiles(user_id: str = Depends(get_current_user)):
    profiles = []
    cursor = models_master_collection.find({"user_id": user_id})
    async for doc in cursor:
        usage_stats = doc.get("usage_stats", {
            "total_uses": 0,
            "successful_uses": 0,
            "success_rate": 0,
            "last_used": None
        })
        created_at = doc.get("created_at")
        if isinstance(created_at, datetime.datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = created_at  # could be None or already a string
        profiles.append({
            "id": str(doc.get("_id")),
            "name": doc.get("name"),
            "description": doc.get("description"),
            "classifier": doc.get("classifier"),
            "embeddings": doc.get("embeddings"),
            "emotion": doc.get("emotion"),
            "toxic": doc.get("toxic"),
            "llama": doc.get("llama"),
            "created_at": created_at_str,
            "usage_stats": usage_stats
        })
    return JSONResponse(content=profiles)

@router.get("/profile-usage-history/{profile_id}")
async def get_profile_usage_history(
    profile_id: str,
    limit: int = 10,
    user_id: str = Depends(get_current_user)
):
    """Get usage history for a specific profile"""
    try:
        # Verify profile belongs to user
        profile = await models_master_collection.find_one({
            "_id": ObjectId(profile_id),
            "user_id": user_id
        })
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Get usage history
        history = await profile_usage_collection.find(
            {"profile_id": ObjectId(profile_id)}
        ).sort("timestamp", -1).limit(limit).to_list(None)
        
        return [{
            "id": str(usage["_id"]),
            "prompt": usage["prompt"],
            "variables": usage["variables"],
            "status": usage["status"],
            "timestamp": usage["timestamp"].isoformat(),
            "completed_at": usage.get("completed_at", "").isoformat() if usage.get("completed_at") else None,
            "error": usage.get("error")
        } for usage in history]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching profile usage history: {str(e)}"
        )

@router.get("/profile-statistics/{profile_id}")
async def get_profile_statistics(
    profile_id: str,
    user_id: str = Depends(get_current_user)
):
    """Get detailed statistics for a specific profile"""
    try:
        # Verify profile belongs to user
        profile = await models_master_collection.find_one({
            "_id": ObjectId(profile_id),
            "user_id": user_id
        })
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Get detailed statistics
        stats = await profile_usage_collection.aggregate([
            {"$match": {"profile_id": ObjectId(profile_id)}},
            {"$group": {
                "_id": None,
                "total_uses": {"$sum": 1},
                "successful_uses": {
                    "$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}
                },
                "failed_uses": {
                    "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                },
                "avg_response_time": {
                    "$avg": {
                        "$subtract": ["$completed_at", "$timestamp"]
                    }
                },
                "first_used": {"$min": "$timestamp"},
                "last_used": {"$max": "$timestamp"}
            }}
        ]).to_list(None)
        
        if not stats:
            return {
                "total_uses": 0,
                "successful_uses": 0,
                "failed_uses": 0,
                "success_rate": 0,
                "avg_response_time": 0,
                "first_used": None,
                "last_used": None
            }
        
        stats = stats[0]
        return {
            "total_uses": stats["total_uses"],
            "successful_uses": stats["successful_uses"],
            "failed_uses": stats["failed_uses"],
            "success_rate": (stats["successful_uses"] / stats["total_uses"] * 100) if stats["total_uses"] > 0 else 0,
            "avg_response_time": stats["avg_response_time"],
            "first_used": stats["first_used"].isoformat() if stats["first_used"] else None,
            "last_used": stats["last_used"].isoformat() if stats["last_used"] else None
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching profile statistics: {str(e)}"
        )

@router.post("/chunk-and-embed-document")
async def chunk_and_embed_document_api(
    document: str = Body(..., description="The document to chunk and embed (plain text or HTML)."),
    is_html: bool = Body(False, description="Whether the document is HTML."),
    chunk_size: int = Body(500, description="Chunk size (characters)."),
    chunk_overlap: int = Body(100, description="Chunk overlap (characters)."),
    models_profile_id: str = Body(..., description="Model profile ID to use for embedding model."),
    user_id: str = Depends(get_current_user)
):
    """
    Chunk a long document and embed the chunks using the selected model profile's embedding model.
    Returns the list of chunks and a success message.
    """
    # Fetch the model profile
    models_profile = await processing_profiles_collection.find_one({"_id": ObjectId(models_profile_id)})
    if not models_profile:
        # fallback to master
        profiles = await models_master_collection.find().to_list(None)
        models_profile = profiles[1] if len(profiles) > 1 else profiles[0]
    analyzer = ArabicRagAnalyzer(None, None, None, models_profile)
    vectorstore, chunks = analyzer.chunk_and_embed_document(document, is_html=is_html, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return {"chunks": chunks, "num_chunks": len(chunks), "success": True} 
