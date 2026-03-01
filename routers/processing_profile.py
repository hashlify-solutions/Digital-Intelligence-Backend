from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from config.db import processing_profiles_collection, models_repository_collection
from utils.auth import get_current_user  # Import JWT auth
import datetime
from schemas.processing_profile import ProcessingProfile 
from schemas.model_repository import ModelRepository
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from typing import Dict, List


processing_router = APIRouter()

@processing_router.post("/", response_model=dict)
async def create_profile(profile: ProcessingProfile, user_id: str = Depends(get_current_user)):
    profile_data = profile.model_dump()
    profile_data["user_id"] = ObjectId(user_id)  
    profile_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    result = await processing_profiles_collection.insert_one(profile_data)
    return {"_id": str(result.inserted_id)}


# 🔍 Get all profiles for a user
@processing_router.get("/", response_model=list)
async def get_profiles(user_id: str = Depends(get_current_user)):
  try:
    profiles = await processing_profiles_collection.find({"user_id": ObjectId(user_id)}).to_list(None)
    profiles_data = []
    for data in profiles:
        data["_id"]= str(data["_id"])
        data["user_id"] = str(data["user_id"])
        profiles_data.append(data)
    return JSONResponse(
      content=profiles_data,
      status_code=200,
    )
  except Exception as e:
    print(f"Exception: {e}")
    raise HTTPException(status_code=500, detail="Some thing went wrong")


# 🔍 Get a single profile by ID
@processing_router.get("/{profile_id}", response_model=dict)
async def get_profile(profile_id: str, user_id: str = Depends(get_current_user)):
    profile = await processing_profiles_collection.find_one({"_id": ObjectId(profile_id), "user_id": ObjectId(user_id)})

    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")

    profile["_id"] = str(profile["_id"])
    profile["user_id"] = str(profile["user_id"])
    return profile


# ✏️ Update a profile
@processing_router.put("/{profile_id}", response_model=dict)
async def update_profile(profile_id: str, profile: ProcessingProfile, user_id: str = Depends(get_current_user)):
    profile_data = profile.model_dump()

    result = await processing_profiles_collection.update_one(
        {"_id": ObjectId(profile_id), "user_id": ObjectId(user_id)},
        {"$set": profile_data}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {"message": "Profile updated successfully"}


# ❌ Delete a profile
@processing_router.delete("/{profile_id}", response_model=dict)
async def delete_profile(profile_id: str, user_id: str = Depends(get_current_user)):
    result = await processing_profiles_collection.delete_one({"_id": ObjectId(profile_id), "user_id": ObjectId(user_id)})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Profile not found")

    return {"message": "Profile deleted successfully"}

@processing_router.post("/model-repository", response_model=dict)
async def create_model(model: ModelRepository, user_id: str = Depends(get_current_user)):
    model_data = model.model_dump()
    model_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    
    if model_data["base_type"] != "embeddings":
        model_data.pop("embedding_size", None)
    
    result = await models_repository_collection.insert_one(model_data)
    return {"_id": str(result.inserted_id)}

@processing_router.get("/model-repository/grouped", response_model=Dict[str, List[dict]])
async def get_models_grouped(user_id: str = Depends(get_current_user)):
    try:
        models = await models_repository_collection.find({"base_type": {"$ne": "entity"}}).to_list(None)
        
        grouped_models = {}
        for model in models:
            model["_id"] = str(model["_id"])
            base_type = model["base_type"]
            if base_type not in grouped_models:
                grouped_models[base_type] = []
            grouped_models[base_type].append(model)
            
        return JSONResponse(
            content=grouped_models,
            status_code=200,
        )
    except Exception as e:
        print(f"Exception: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong")