import pandas as pd
import logging
import datetime
import chardet
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from config.db import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from rag import ArabicRagAnalyzer
from typing import Dict
from utils.helpers import create_quadrant_collection_if_not_exists, robust_qdrant_upsert

# Configure logging
logging.basicConfig(
    filename="csv_ingestion.log",  # Logs are saved to this file
    level=logging.INFO,  # Logs INFO level and above
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class CSVIngester:
    def __init__(self, 
                mongo_collection: AsyncIOMotorCollection, 
                mongo_collection_all_cases: AsyncIOMotorCollection, 
                case_id: str, 
                models_profile: Dict = None):
        self.mongo_collection = mongo_collection
        self.collection_all_cases = mongo_collection_all_cases
        self.case_id = case_id
        self.models_profile = models_profile

    async def ingest_csv_manually(self, file_path):
      """Manually ingest CSV data into MongoDB."""
      try:
          logger.info("Starting manual CSV ingestion...")

          await self.collection_all_cases.find_one_and_update(
            {"_id": ObjectId(self.case_id)},
            {
                "$set": {
                    "status": "injestion_started",
                    "injesting_started_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
            }
          )

          with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']

          # Load CSV into DataFrame
          logger.info(f"Loading data from {file_path}...")
          data = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')

          # Log basic information about the data
          logger.info(f"Number of rows in CSV: {len(data)}")
          logger.info(f"Columns in CSV: {list(data.columns)}")

          # Convert DataFrame to list of dictionaries
          logger.info("Converting data to dictionary format...")
          data_dict = data.to_dict(orient="records")

          # Insert data into MongoDB
          logger.info(f"Inserting {len(data_dict)} rows into MongoDB...")
          inserted_count = 0
          skipped_count = 0

          for record in data_dict:
              try:
                  if record.get("Preview Text") is not None and pd.notna(record.get("Preview Text")):
                    # Check if the document already exists in MongoDB
                    case_item = await self.mongo_collection.find_one({"_id": ObjectId(record.get("_id"))})
                    if not case_item:
                        # Add a timestamp to the record
                        record["ingestion_timestamp"] = datetime.datetime.now().isoformat()
                        record["case_id"] = ObjectId(self.case_id)
                        record["processed"] = False
                        record["alert"] = None
                        # Insert the record into MongoDB
                        await self.mongo_collection.insert_one(record)
                        inserted_count += 1
                    else:
                        logger.warning(f"Skipping duplicate document with _id: {record.get('_id')}")
                        skipped_count += 1
              except Exception as e:
                  logger.error(f"Error inserting document: {e}\nDocument: {record}")


          logger.info(f"Successfully inserted {inserted_count} rows into MongoDB!")
          logger.info(f"Skipped {skipped_count} duplicate rows.")

          await self.collection_all_cases.find_one_and_update(
            {"_id": ObjectId(self.case_id)},
            {
                "$set": {
                    "status": "injestion_completed",
                    "injesting_completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
                }
            }
          )
          # set toal insert count
          await self.collection_all_cases.find_one_and_update(
            {"_id": ObjectId(self.case_id)},
            {
                "$set": {
                    "total_messages": inserted_count
                }
            }
          )

      except Exception as e:
          logger.error(f"Error during ingestion: {e}")
    
    async def ingest_rag_data(self):
        try:
            logger.info("::Entering in ingest_rag_data function::")
            logger.info("Starting manual RAG ingestion...")
            
            embedding_obj = self.models_profile.get("embeddings", {})
            vector_size = embedding_obj.get("embedding_size", 512)

            data = await self.mongo_collection.find().to_list(None)
            
            logger.info(f"Number of rows of the case messages in dB are: {len(data)}")
            
            analyzer = ArabicRagAnalyzer(self.collection_all_cases, self.mongo_collection, self.case_id, self.models_profile)
            
            new_quadrant_case_collection = f"case_{self.case_id}"
            create_quadrant_collection_if_not_exists(new_quadrant_case_collection, vector_size, Distance.COSINE)
                
            prepared_data = []
            logger.info(f"Preparing data for ingestion for injestion in quadrant")
            for message in data:
                points = analyzer.query_embedding_endpoint(message.get("Preview Text"))
                if points:
                    logger.info(f"Points exist for {message.get("_id")} message: {points}")
                    prepared_data.append(PointStruct(
                        id=str(uuid.uuid4()),
                        payload = {
                            "message": message.get("Preview Text"),
                            "mongo_id": str(message.get("_id")),
                            "from": message.get("From"),
                            "to": message.get("To"),
                            "date": message.get("Date"),
                            "case_name": message.get("Case Name")
                        }, 
                        vector=points
                    ))
                else:
                    logger.info(f"Points does not exist for {message.get("_id")} message")
            
            success = robust_qdrant_upsert(
                collection_name=new_quadrant_case_collection,
                points=prepared_data,
                max_retries=3
            )
            if success:
                logger.info(f"Successfully ingested {len(prepared_data)} points to collection {new_quadrant_case_collection}")
            else:
                logger.error(f"Failed to ingest points to collection {new_quadrant_case_collection} after multiple attempts")             
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")

# Example usage
# file_path = "/Users/khaledhegazey/Downloads/DIS English /Cleaned_sample_data.csv"  # Update with your CSV file path
# ingest_csv_manually(file_path)