from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pathlib import Path
import os
import mimetypes
from typing import Optional
from utils.case_mapping import get_actual_case_directory

router = APIRouter()

@router.get("/files/{case_name:path}/{file_path:path}")
async def serve_media_file(case_name: str, file_path: str):
    """
    Serve media files from UFDR exhibits
    
    Args:
        case_name: Name of the case/exhibit (e.g., "Exhibit GG1 - Apple iPad Air Method 1")
        file_path: Path to the file within the case (e.g., "files/Image/3507_19.JPG")
    """
    try:
        # Construct the full file path
        base_path = Path(".")
        
        # Log for debugging
        print(f"DEBUG: Serving file - case_name: {case_name}, file_path: {file_path}")
        
        # Normalize the path construction
        # The URL already includes the correct path structure
        full_path = base_path / case_name / file_path
        
        print(f"DEBUG: Full path constructed: {full_path}")
        print(f"DEBUG: Absolute path: {full_path.resolve()}")
        print(f"DEBUG: File exists: {full_path.exists()}")
        
        # Security check - ensure the path is within the project directory
        full_path = full_path.resolve()
        project_root = base_path.resolve()
        
        # Allow access to files within case directories
        case_dir = (base_path / case_name).resolve()
        if not str(full_path).startswith(str(case_dir)):
            print(f"DEBUG: Security check failed - path outside case directory")
            raise HTTPException(status_code=403, detail="Access denied")
            
        # Check if file exists
        if not full_path.exists():
            print(f"DEBUG: File not found at path: {full_path}")
            raise HTTPException(status_code=404, detail="File not found")
            
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail="Path is not a file")
            
        # Determine content type
        content_type, _ = mimetypes.guess_type(str(full_path))
        if not content_type:
            content_type = "application/octet-stream"
            
        return FileResponse(
            path=str(full_path),
            media_type=content_type,
            filename=full_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@router.get("/cases/{case_id}/media/{file_path:path}")
async def serve_media_by_case_id(case_id: str, file_path: str):
    """
    Serve media files using case ID and file path
    
    Args:
        case_id: Case ID from the database
        file_path: Relative path to the file
    """
    try:
        # Map case ID to case name (this should be enhanced to query database)
        # For now, use the case_id as the case name and let auto-detection handle it
        case_name = case_id
        
        # Use auto-detection to find the actual directory
        actual_case_dir = get_actual_case_directory(case_name)
        
        # Verify the directory exists
        base_path = Path(".")
        case_dir_path = base_path / actual_case_dir
        if not case_dir_path.exists():
            raise HTTPException(status_code=404, detail=f"Case directory not found: {actual_case_dir}")
            
        return await serve_media_file(actual_case_dir, file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@router.get("/health")
async def media_health_check():
    """Health check endpoint for media serving"""
    return {"status": "healthy", "service": "media_files"}