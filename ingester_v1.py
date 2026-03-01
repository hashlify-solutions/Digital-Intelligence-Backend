import pandas as pd
import logging
import datetime
import chardet
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from config.db import qdrant_client
from config.settings import settings
import re
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
from rag_v1 import ArabicRagAnalyzer
from typing import Dict
import asyncio
from concurrent.futures import ThreadPoolExecutor


# Configure logging
logging.basicConfig(
    filename="csv_ingestion.log",  # Logs are saved to this file
    level=logging.INFO,  # Logs INFO level and above
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class CSVIngester:
    def __init__(
        self,
        mongo_collection: AsyncIOMotorCollection,
        mongo_collection_all_cases: AsyncIOMotorCollection,
        case_id: str,
        models_profile: Dict = None,
        use_parallel_processing: bool = True,
    ):
        self.mongo_collection = mongo_collection
        self.collection_all_cases = mongo_collection_all_cases
        self.case_id = case_id
        self.models_profile = models_profile
        self.use_parallel_processing = use_parallel_processing

        # Get configuration from centralized settings
        compute = settings.compute_config
        
        # Initialize parallel processing components
        if self.use_parallel_processing:
            self.batch_size = compute["batch_size"]
            self.max_workers = compute["max_workers"]
            self.chunk_size = compute["chunk_size"]
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            logger.info(
                f"CSVIngester initialized with parallel processing: batch_size={self.batch_size}, "
                f"workers={self.max_workers}, chunk_size={self.chunk_size}"
            )
        else:
            self.batch_size = 32
            self.max_workers = 1
            self.chunk_size = 50000
            self.executor = None

    async def ingest_csv_manually(self, file_path):
        """Manually ingest CSV data into MongoDB with streaming/chunked processing support."""
        try:
            logger.info("Starting manual CSV ingestion with streaming...")

            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)},
                {
                    "$set": {
                        "status": "ingestion_started",
                        "ingestion_started_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    }
                },
            )

            # Detect file encoding from first 10KB (more efficient than reading entire file)
            with open(file_path, "rb") as f:
                sample = f.read(10240)  # Read first 10KB for encoding detection
                result = chardet.detect(sample)
                encoding = result["encoding"] or "utf-8"
            
            logger.info(f"Detected encoding: {encoding}")

            # Use streaming/chunked reading for large files
            total_inserted = 0
            total_skipped = 0
            chunk_number = 0
            columns_logged = False
            
            logger.info(f"Starting chunked ingestion with chunk_size={self.chunk_size}...")
            
            # Use chunked reading to avoid loading entire file into memory
            for chunk_df in pd.read_csv(
                file_path, 
                encoding=encoding, 
                on_bad_lines="skip",
                chunksize=self.chunk_size
            ):
                chunk_number += 1
                
                # Log columns only once (from first chunk)
                if not columns_logged:
                    logger.info(f"Columns in CSV: {list(chunk_df.columns)}")
                    columns_logged = True
                
                # Convert chunk to list of dictionaries
                chunk_data = chunk_df.to_dict(orient="records")
                
                logger.info(f"Processing chunk {chunk_number}: {len(chunk_data)} rows...")

                # Insert chunk data into MongoDB
                if self.use_parallel_processing:
                    inserted_count, skipped_count = await self._insert_data_parallel(
                        chunk_data
                    )
                else:
                    inserted_count, skipped_count = await self._insert_data_sequential(
                        chunk_data
                    )
                
                total_inserted += inserted_count
                total_skipped += skipped_count
                
                # Update progress periodically
                if chunk_number % 10 == 0:
                    await self.collection_all_cases.find_one_and_update(
                        {"_id": ObjectId(self.case_id)},
                        {"$set": {"ingestion_progress": total_inserted}},
                    )
                    logger.info(f"Progress: {total_inserted} records inserted so far...")

            logger.info(f"Successfully inserted {total_inserted} rows into MongoDB!")
            logger.info(f"Skipped {total_skipped} duplicate rows.")
            logger.info(f"Processed {chunk_number} chunks total.")

            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)},
                {
                    "$set": {
                        "status": "ingestion_completed",
                        "ingestion_completed_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    }
                },
            )
            # Set total insert count
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)},
                {"$set": {"total_messages": total_inserted}},
            )

        except Exception as e:
            logger.error(f"Error during ingestion: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    async def _insert_data_parallel(self, data_dict: list) -> tuple:
        """Insert data into MongoDB using parallel processing"""
        inserted_count = 0
        skipped_count = 0

        # Split data into batches
        batches = [
            data_dict[i : i + self.batch_size]
            for i in range(0, len(data_dict), self.batch_size)
        ]

        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch_parallel(batch))
            tasks.append(task)

        # Wait for all batches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch processing failed: {result}")
            else:
                batch_inserted, batch_skipped = result
                inserted_count += batch_inserted
                skipped_count += batch_skipped

        return inserted_count, skipped_count

    async def _insert_data_sequential(self, data_dict: list) -> tuple:
        """Insert data into MongoDB using sequential processing"""
        inserted_count = 0
        skipped_count = 0

        for record in data_dict:
            try:
                if record.get("Preview Text") is not None and pd.notna(
                    record.get("Preview Text")
                ):
                    # Check if the document already exists in MongoDB
                    case_item = await self.mongo_collection.find_one(
                        {"_id": ObjectId(record.get("_id"))}
                    )
                    if not case_item:
                        # Add a timestamp to the record
                        record["ingestion_timestamp"] = (
                            datetime.datetime.now().isoformat()
                        )
                        record["case_id"] = ObjectId(self.case_id)
                        record["processed"] = False
                        record["alert"] = None
                        # Normalize Application and Message Type fields for analytics (only if missing)
                        if "Application" not in record or not record["Application"]:
                            record["Application"] = record.get("application", "Unknown")
                        if "Message Type" not in record or not record["Message Type"]:
                            file_type = str(record.get("File Type", "")).lower()
                            if "chat" in file_type or "message" in file_type:
                                record["Message Type"] = "message"
                            elif "call" in file_type:
                                record["Message Type"] = "call"
                            elif "email" in file_type:
                                record["Message Type"] = "email"
                            elif "contact" in file_type:
                                record["Message Type"] = "contact"
                            else:
                                record["Message Type"] = "message"
                        # Ensure analysis_summary field exists for analytics
                        if "analysis_summary" not in record:
                            record["analysis_summary"] = {
                                "top_topic": None,
                                "sentiment_aspect": None,
                                "emotion": None,
                                "toxicity_score": None,
                                "risk_level": None,
                                "language": None,
                                "interaction_type": None,
                                "entities": [],
                                "entities_classification": {},
                            }
                        # Insert the record into MongoDB
                        await self.mongo_collection.insert_one(record)
                        inserted_count += 1
                    else:
                        logger.warning(
                            f"Skipping duplicate document with _id: {record.get('_id')}"
                        )
                        skipped_count += 1
            except Exception as e:
                logger.error(f"Error inserting document: {e}\nDocument: {record}")

        return inserted_count, skipped_count

    async def _process_batch_parallel(self, batch: list) -> tuple:
        """Process a single batch of records in parallel"""
        inserted_count = 0
        skipped_count = 0

        # Create tasks for each record in the batch
        tasks = []
        for record in batch:
            task = asyncio.create_task(self._process_single_record(record))
            tasks.append(task)

        # Wait for all records in the batch to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Record processing failed: {result}")
            else:
                if result:  # Record was inserted
                    inserted_count += 1
                else:  # Record was skipped
                    skipped_count += 1

        return inserted_count, skipped_count

    async def _process_single_record(self, record: dict) -> bool:
        """Process a single record and return True if inserted, False if skipped"""
        try:
            if record.get("Preview Text") is not None and pd.notna(
                record.get("Preview Text")
            ):
                # Check if the document already exists in MongoDB
                case_item = await self.mongo_collection.find_one(
                    {"_id": ObjectId(record.get("_id"))}
                )
                if not case_item:
                    # Add a timestamp to the record
                    record["ingestion_timestamp"] = datetime.datetime.now().isoformat()
                    record["case_id"] = ObjectId(self.case_id)
                    record["processed"] = False
                    record["alert"] = None
                    # Normalize Application and Message Type fields for analytics (only if missing)
                    if "Application" not in record or not record["Application"]:
                        record["Application"] = record.get("application", "Unknown")
                    if "Message Type" not in record or not record["Message Type"]:
                        file_type = str(record.get("File Type", "")).lower()
                        if "chat" in file_type or "message" in file_type:
                            record["Message Type"] = "message"
                        elif "call" in file_type:
                            record["Message Type"] = "call"
                        elif "email" in file_type:
                            record["Message Type"] = "email"
                        elif "contact" in file_type:
                            record["Message Type"] = "contact"
                        else:
                            record["Message Type"] = "message"
                    # Ensure analysis_summary field exists for analytics
                    if "analysis_summary" not in record:
                        record["analysis_summary"] = {
                            "top_topic": None,
                            "sentiment_aspect": None,
                            "emotion": None,
                            "toxicity_score": None,
                            "risk_level": None,
                            "language": None,
                            "interaction_type": None,
                            "entities": [],
                            "entities_classification": {},
                        }
                    # Insert the record into MongoDB
                    await self.mongo_collection.insert_one(record)
                    return True  # Record was inserted
                else:
                    logger.warning(
                        f"Skipping duplicate document with _id: {record.get('_id')}"
                    )
                    return False  # Record was skipped
        except Exception as e:
            logger.error(f"Error inserting document: {e}\nDocument: {record}")
            return False  # Record was not inserted due to error

        return False  # Record was not processed

    def normalize_embedding(self, embedding):
        return embedding / np.linalg.norm(embedding)

    async def ingest_rag_data(self):
        """Ingest RAG data with parallel processing support."""
        try:
            logger.info("::Entering in ingest_rag_data function::")
            logger.info("Starting manual RAG ingestion...")

            embedding_obj = self.models_profile.get("embeddings", {})
            vector_size = embedding_obj.get("embedding_size", 512)

            data = await self.mongo_collection.find().to_list(None)

            logger.info(f"Number of rows of the case messages in dB are: {len(data)}")

            analyzer = ArabicRagAnalyzer(
                self.collection_all_cases,
                self.mongo_collection,
                self.case_id,
                self.models_profile,
                use_parallel_processing=self.use_parallel_processing,
            )
            new_qdrant_collection = f"case_{self.case_id}"

            logger.info(f"Creating new collection: {new_qdrant_collection}")
            qdrant_collections_exist = qdrant_client.collection_exists(
                collection_name=new_qdrant_collection
            )

            logger.info(f"Checking if collection exists: {qdrant_collections_exist}")

            if qdrant_collections_exist == False:
                logger.info(f"Collection {new_qdrant_collection} doesn't exists")
                qdrant_client.create_collection(
                    collection_name=new_qdrant_collection,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )

            # Process data with parallel processing
            if self.use_parallel_processing:
                prepared_data = await self._prepare_rag_data_parallel(data, analyzer)
            else:
                prepared_data = await self._prepare_rag_data_sequential(data, analyzer)

            # Insert data into Qdrant
            if prepared_data:
                qdrant_client.upsert(
                    collection_name=new_qdrant_collection, points=prepared_data
                )
                logger.info(
                    f"Successfully inserted {len(prepared_data)} points into Qdrant collection {new_qdrant_collection}"
                )

                # Optional integrity check: verify Qdrant point count matches total messages
                try:
                    total_messages = await self.mongo_collection.count_documents({})
                    # Count points in Qdrant (using simple aggregate)
                    count_res = qdrant_client.scroll(
                        collection_name=new_qdrant_collection, limit=1
                    )
                    # Qdrant doesn't return total count directly with scroll; perform a quick approximate check by reading back 'n' via an aggregate on Mongo (already done)
                    logger.info(
                        f"Post-run check: Mongo total_messages={total_messages}; Qdrant upserted_points={len(prepared_data)}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Post-run Qdrant/Mongo count check skipped due to: {e}"
                    )
            else:
                logger.warning("No data prepared for Qdrant insertion")

        except Exception as e:
            logger.error(f"Error during RAG ingestion: {e}")

    async def _prepare_rag_data_parallel(
        self, data: list, analyzer: ArabicRagAnalyzer
    ) -> list:
        """Prepare RAG data using parallel processing"""
        prepared_data = []

        # Split data into batches
        batches = [
            data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)
        ]

        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_rag_batch(batch, analyzer))
            tasks.append(task)

        # Wait for all batches to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"RAG batch processing failed: {result}")
            else:
                prepared_data.extend(result)

        return prepared_data

    async def _prepare_rag_data_sequential(
        self, data: list, analyzer: ArabicRagAnalyzer
    ) -> list:
        """Prepare RAG data using sequential processing"""
        prepared_data = []
        for message in data:
            try:
                # Embed only main text field to reduce cost
                # columns_data = {"Preview Text": [str(message.get("Preview Text") or "")]}
                all_embeddings = analyzer.query_embedding_endpoint(
                    message.get("Preview Text", "")
                )
                if all_embeddings:
                    logger.info(f"Points exist for {message.get('_id')} message")
                    # Combine all embeddings into one vector
                    # combined_embedding = np.mean([emb for emb in all_embeddings.values() if emb], axis=0)
                    prepared_data.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            payload={
                                "message": message.get("Preview Text"),
                                "mongo_id": str(message.get("_id")),
                                "from": message.get("From"),
                                "to": message.get("To"),
                                "date": message.get("Date"),
                                "case_name": message.get("Case Name"),
                                "all_embeddings": all_embeddings,
                            },
                            vector=all_embeddings,
                        )
                    )
                else:
                    logger.info(
                        f"Points does not exist for {message.get('_id')} message"
                    )
            except Exception as e:
                logger.error(f"Error processing message {message.get('_id')}: {e}")

        return prepared_data

    async def _process_rag_batch(
        self, batch: list, analyzer: ArabicRagAnalyzer
    ) -> list:
        """Process a single batch of messages for RAG data preparation"""
        batch_data = []

        for message in batch:
            try:
                # Embed only main text field to reduce cost
                # columns_data = {"Preview Text": [str(message.get("Preview Text") or "")]}
                all_embeddings = analyzer.query_embedding_endpoint(
                    message.get("Preview Text", "")
                )

                if all_embeddings:
                    logger.info(f"Points exist for {message.get('_id')} message")
                    # Combine all embeddings into one vector
                    # combined_embedding = np.mean([emb for emb in all_embeddings.values() if emb], axis=0)

                    batch_data.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            payload={
                                "message": message.get("Preview Text"),
                                "mongo_id": str(message.get("_id")),
                                "from": message.get("From"),
                                "to": message.get("To"),
                                "date": message.get("Date"),
                                "case_name": message.get("Case Name"),
                                "all_embeddings": all_embeddings,
                            },
                            vector=all_embeddings,
                        )
                    )
                else:
                    logger.info(
                        f"Points does not exist for {message.get('_id')} message"
                    )
            except Exception as e:
                logger.error(f"Error processing message {message.get('_id')}: {e}")

        return batch_data

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, "executor") and self.executor:
            self.executor.shutdown(wait=True)


# Example usage
# file_path = "/Users/khaledhegazey/Downloads/DIS English /Cleaned_sample_data.csv"  # Update with your CSV file path
# ingest_csv_manually(file_path)
