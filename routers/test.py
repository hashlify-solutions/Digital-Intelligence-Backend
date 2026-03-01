from fastapi import APIRouter, HTTPException, Depends, Body
from bson import ObjectId
from typing import List
from rag import ArabicRagAnalyzer
from ingester import CSVIngester
from config.db import collection_case, db
from analyzer import ArabicSocialAnalyzer





router = APIRouter()

# Signup Route
@router.post("/create-embeddings")
async def create_embeddings():
  try:
    id = "67c8b1d85c51a98c42b48452" #arabic
    # id = "67c5ae44cdf949a6b1ff8cd0" #english
    case = await collection_case.find_one({"_id": ObjectId(id)})
    collection = db[f"{case['name']}_{id}"]
    ingester = CSVIngester(collection, collection_case, id, case.get("model_profile"))
    await ingester.ingest_rag_data()
    return {"message": "Embeddings created successfully."}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error creating embeddings: {e}")
    
@router.post("/semantic-search")
async def semantic_search(text: str):
  try:
    id = "67c8b1d85c51a98c42b48452" #arabic
    # id = "67c5ae44cdf949a6b1ff8cd0" #english
    case = await collection_case.find_one({"_id": ObjectId(id)})
    collection = db[f"{case['name']}_{id}"]
    analyzer = ArabicRagAnalyzer(collection, collection_case, id, case.get("model_profile"))
    results = analyzer.semantic_search(query=text, collection_name=f"case_{id}", limit=3, score_threshold=0.15)
    return {"results": results}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error searching for similar documents: {e}")
  
@router.post("/analyze-user-rag-query")
async def analyze_user_rag_query(query: str):
  try:
    id = "67c8b1d85c51a98c42b48452" #arabic
    # id = "67c5ae44cdf949a6b1ff8cd0" #english
    case = await collection_case.find_one({"_id": ObjectId(id)})
    collection = db[f"{case['name']}_{id}"]
    analyzer = ArabicRagAnalyzer(collection, collection_case, id, case.get("model_profile"))
    summary = analyzer.summarize_messages(query)
    if summary is None or summary.get("success") == False:
      raise HTTPException(status_code=404, detail="Failed to generate summary.")
    return summary.get("summary")
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error analyzing user query: {e}")
  
  
@router.post("/extract-entities")
async def extract_entities(text: str):
  try:
    id = "67c8b1d85c51a98c42b48452" #arabic
    # id = "67c5ae44cdf949a6b1ff8cd0" #english
    case = await collection_case.find_one({"_id": ObjectId(id)})
    collection = db[f"{case['name']}_{id}"]
    analyzer = ArabicSocialAnalyzer(collection, collection_case, id, case.get("model_profile"))
    entities = analyzer.extract_entities(text)
    return {"entities": entities}
  except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error analyzing user query: {e}")
