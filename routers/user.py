from fastapi import APIRouter, HTTPException, Depends, Body
from bson import ObjectId
from config.settings import settings
from schemas.user import UserCreate, UserOut, UserLogin
from config.db import users_collection
from utils.auth import hash_password, verify_password, create_access_token, decode_access_token
from datetime import timedelta
from typing import List
from utils.auth import get_current_user  # Import JWT auth


router = APIRouter()

# Signup Route
@router.post("/signup", response_model=UserOut)
async def create_user(user: UserCreate):
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pwd = hash_password(user.password)
    new_user = {"name": user.name, "email": user.email, "password": hashed_pwd}
    result = await users_collection.insert_one(new_user)

    return {**new_user, "id": str(result.inserted_id), "password": None}

# Login Route
@router.post("/login")
async def login(user: UserLogin):
    existing_user = await users_collection.find_one({"email": user.email})
    if not existing_user or not verify_password(user.password, existing_user["password"]):
        raise HTTPException(status_code=400, detail="Invalid credentials")

    token = create_access_token({"user_id": str(existing_user["_id"])}, timedelta(minutes=settings.access_token_expire_minutes))
    refresh_token = create_access_token({"user_id": str(existing_user["_id"])}, timedelta(minutes=settings.refresh_token_expire_minutes))
    return {"access_token": token, "token_type": "bearer", "refresh_token":refresh_token}

# Get User by ID
@router.get("/get-by-token", response_model=UserOut)
async def get_user(user_id: str = Depends(get_current_user)):
    user = await users_collection.find_one({"_id": ObjectId(user_id)}, {"password": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {**user, "id": str(user["_id"])}

# Get All Users
@router.get("/", response_model=List[UserOut])
async def get_all_users(_: str = Depends(get_current_user)):
    users = await users_collection.find({}, {"password": 0}).to_list(100)
    return [{**user, "id": str(user["_id"])} for user in users]

#refresh token
@router.post("/refresh-token")
async def refresh_token(data: dict = Body(...)):
    try:
        token = data.get("token")
        payload = decode_access_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        token = create_access_token({"user_id": payload["user_id"]}, timedelta(minutes=30))
        refresh_token = create_access_token({"user_id": payload["user_id"]}, timedelta(days=7))
        return {"access_token": token, "token_type": "bearer", "refresh_token":refresh_token}
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")