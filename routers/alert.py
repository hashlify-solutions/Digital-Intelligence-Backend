from fastapi import APIRouter, Depends, HTTPException
from bson import ObjectId
from schemas.alert import AlertModel
from config.db import alerts_collection
from utils.auth import get_current_user  # Import JWT auth
import datetime


alert_router = APIRouter()

# 🚀 Create an alert (Authenticated)
@alert_router.post("/", response_model=dict)
async def create_alert(alert: AlertModel, user_id: str = Depends(get_current_user)):
    alert_data = alert.model_dump()
    alert_data["user_id"] = ObjectId(user_id)  # Ensure user_id is an ObjectId
    alert_data["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = await alerts_collection.insert_one(alert_data)
    return {"_id": str(result.inserted_id)}

# 🔍 Get alerts for the logged-in user
@alert_router.get("/", response_model=list)
async def get_user_alerts(user_id: str = Depends(get_current_user)):
    pipeline = [
    {
        "$match": {"user_id": ObjectId(user_id)}
    },
    {
        "$lookup": {
            "from": "Users",  # Name of the users collection
            "localField": "user_id",
            "foreignField": "_id",
            "as": "user"
        }
    },
    {
        "$unwind": "$user"  # Convert array to object if a user is found
    },
    {
        "$project": {
            "_id": {"$toString": "$_id"},  # Convert _id to string
            "toxicity_score": 1,
            "risk_level": 1,
            "sentiment_aspect": 1,
            "emotion": 1,
            "language": 1,
            "top_topic": 1,
            "interaction_type": 1,
            "entities": 1,
            "user": {
                "_id": {"$toString": "$user._id"},
                "name": "$user.name",
                "email": "$user.email"
            },
            "name": 1,
            "description": 1,
            "created_at": 1
        }
    }
    ]

    alerts_cursor = alerts_collection.aggregate(pipeline)
    alerts = await alerts_cursor.to_list(length=None)
    return alerts

# 📌 Get specific alert by ID (only if the user owns it) 
@alert_router.get("/{alert_id}", response_model=dict)
async def get_alert(alert_id: str, user_id: str = Depends(get_current_user)):
    print(alert_id, user_id)
    alert = await alerts_collection.find_one({"_id": ObjectId(alert_id), "user_id": ObjectId(user_id)})
    print(alert)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found or unauthorized")
    alert["_id"] = str(alert["_id"])
    alert["user_id"] = str(alert["user_id"])
    return alert

# ✏️ Update an alert (only if the user owns it)
@alert_router.put("/{alert_id}", response_model=dict)
async def update_alert(alert_id: str, alert: AlertModel, user_id: str = Depends(get_current_user)):
    alert_data = {k: v for k, v in alert.dict().items() if v is not None}
    result = await alerts_collection.update_one(
        {"_id": ObjectId(alert_id), "user_id": ObjectId(user_id)}, {"$set": alert_data}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found or unauthorized")
    return {"message": "Alert updated successfully"}

# ❌ Delete an alert (only if the user owns it)
@alert_router.delete("/{alert_id}", response_model=dict)
async def delete_alert(alert_id: str, user_id: str = Depends(get_current_user)):
    result = await alerts_collection.delete_one({"_id": ObjectId(alert_id), "user_id": ObjectId(user_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Alert not found or unauthorized")
    return {"message": "Alert deleted successfully"}