from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pathlib import Path
import os
import shutil
from datetime import datetime
from config.db import platform_control_collection, users_collection
from bson import ObjectId
from utils.auth import get_current_user

router = APIRouter()

# Base logo directory
LOGO_DIR = Path("logo")
LOGO_DIR.mkdir(exist_ok=True)

@router.post("/update")
async def update_platform_details(
    name: str = Form(...),
    logo: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)  
):     
    try:
        if not logo.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Convert user ID to ObjectId and string for directory name
        user_id = ObjectId(current_user)
        user_id_str = str(user_id)
        
        # Create user-specific logo directory
        user_logo_dir = LOGO_DIR / user_id_str
        user_logo_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_extension = os.path.splitext(logo.filename)[1] if logo.filename else ".png"
        logo_filename = f"platform_logo_{timestamp}{file_extension}"
        logo_path = user_logo_dir / logo_filename
        
        # Save the uploaded file to user's directory
        with open(logo_path, "wb") as buffer:
            shutil.copyfileobj(logo.file, buffer)
        
        # Store the relative path that will be used by the frontend
        relative_logo_path = f"/logo/{user_id_str}/{logo_filename}"
        
        existing_record = await platform_control_collection.find_one({"userId": user_id})
        
        if existing_record:
            await platform_control_collection.update_one(
                {"_id": existing_record["_id"]},
                {"$set": {
                    "name": name,
                    "logo": relative_logo_path,
                    "updated_at": datetime.now()
                }}
            )
            # Delete old logo file if it exists and is different
            if "logo" in existing_record:
                old_logo_path = existing_record["logo"].lstrip("/")
                old_file = Path(old_logo_path)
                if old_file.exists() and str(old_file) != str(logo_path):
                    old_file.unlink()
        else:
            await platform_control_collection.insert_one({
                "userId": user_id,
                "name": name,
                "logo": relative_logo_path,
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            })
        
        return {
            "status": "success",
            "message": "Platform details updated successfully",
            "data": {
                "name": name,
                "logo_path": relative_logo_path
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating platform details: {str(e)}")

@router.get("/details")
async def get_platform_details(current_user: dict = Depends(get_current_user)):
    user_id = ObjectId(current_user)
    platform_details = await platform_control_collection.find_one({"userId": user_id})
    
    if not platform_details:
        return {
            "name": "Digital Intelligence Platform",
            "logo": None,
            "userId": str(user_id)  
        }
    
    platform_details["_id"] = str(platform_details["_id"])
    if "userId" in platform_details and isinstance(platform_details["userId"], ObjectId):
        platform_details["userId"] = str(platform_details["userId"])
    
    return platform_details 