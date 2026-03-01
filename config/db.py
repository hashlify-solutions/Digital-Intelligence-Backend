from motor.motor_asyncio import AsyncIOMotorClient
from qdrant_client import QdrantClient
from config.settings import settings
import logging
import asyncio
logger = logging.getLogger(__name__)
from utils.neo4j_client import Neo4jClient, init_constraints

# Get compute configuration for optimized connection settings
_compute_config = settings.compute_config

# Calculate optimal connection pool sizes based on hardware
# Rule of thumb: max_pool_size should be 2-3x the number of concurrent workers
_max_pool_size = min(_compute_config["max_workers"] * 2, 100)  # Cap at 100
_min_pool_size = min(_compute_config["max_workers"], 10)  # Min connections to maintain

# MongoDB connection with optimized pooling
# These settings are critical for high-throughput processing
client = AsyncIOMotorClient(
    settings.mongo_connection_string,
    maxPoolSize=_max_pool_size,  # Maximum connections in pool
    minPoolSize=_min_pool_size,  # Minimum connections to maintain
    maxIdleTimeMS=60000,  # Close idle connections after 60s
    waitQueueTimeoutMS=30000,  # Wait up to 30s for a connection
    serverSelectionTimeoutMS=30000,  # Timeout for server selection
    connectTimeoutMS=10000,  # Connection timeout
    socketTimeoutMS=300000,  # 5 minute socket timeout for long operations
    retryWrites=True,  # Automatically retry failed writes
    retryReads=True,  # Automatically retry failed reads
    w='majority',  # Write concern for durability
    journal=True,  # Enable journaling for durability
)

logger.info(
    f"MongoDB client initialized with pool: max={_max_pool_size}, min={_min_pool_size}"
)

# Enhanced Qdrant client with proper timeout configuration
try:
    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=60,  # 60 second timeout for operations
        prefer_grpc=False,  # Use HTTP client for better compatibility
    )
    logger.info("Qdrant client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant client: {e}")
    raise
db = client[settings.mongo_database]


# Neo4j Client Initialization Via Lazy singleton initialization to avoid failing app startup if Neo4j is down
_neo: Neo4jClient | None = None
_constraints_inited: bool = False

def get_neo() -> Neo4jClient:
    global _neo, _constraints_inited
    if _neo is None:
        try:
            logger.info(f"Connecting to Neo4j at {settings.neo4j_uri}")
            _neo = Neo4jClient(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password, database=settings.neo4j_database)
            logger.info("Neo4j connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Cannot connect to Neo4j at {settings.neo4j_uri}: {e}")
    if not _constraints_inited:
        try:
            init_constraints(_neo)
            _constraints_inited = True
        except Exception as e:
            logger.warning(f"Could not initialize constraints: {e}")
            # Defer constraint creation if DB isn't reachable yet; endpoint will surface error
            pass
    return _neo

# Centralized DB error handling and retry logic
def with_db_retry(max_retries=3, delay=2):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    logging.warning(f"DB operation failed (attempt {attempt}): {e}")
                    if attempt == max_retries:
                        raise
                    await asyncio.sleep(delay * attempt)
            raise last_exc
        return wrapper
    return decorator


# Optimized bulk write size based on hardware configuration
# For high-memory systems, larger batches are more efficient
_bulk_write_size = min(_compute_config["batch_size"] * 10, 5000)  # Cap at 5000 docs per batch

async def bulk_insert_optimized(collection, documents: list, ordered: bool = False) -> int:
    """
    Perform optimized bulk insert with automatic batching.
    
    Args:
        collection: MongoDB collection to insert into
        documents: List of documents to insert
        ordered: If True, preserves order but stops on error (default: False for parallelism)
        
    Returns:
        Total number of documents inserted
    """
    if not documents:
        return 0
    
    total_inserted = 0
    
    for i in range(0, len(documents), _bulk_write_size):
        batch = documents[i:i + _bulk_write_size]
        try:
            result = await collection.insert_many(batch, ordered=ordered)
            total_inserted += len(result.inserted_ids)
        except Exception as e:
            logger.error(f"Bulk insert error at batch {i // _bulk_write_size}: {e}")
            # Continue with next batch if unordered
            if ordered:
                raise
    
    logger.debug(f"Bulk insert completed: {total_inserted}/{len(documents)} documents")
    return total_inserted


async def bulk_update_optimized(collection, updates: list, ordered: bool = False) -> int:
    """
    Perform optimized bulk update with automatic batching.
    
    Args:
        collection: MongoDB collection to update
        updates: List of (filter, update) tuples
        ordered: If True, preserves order but stops on error (default: False for parallelism)
        
    Returns:
        Total number of documents modified
    """
    from pymongo import UpdateOne
    
    if not updates:
        return 0
    
    total_modified = 0
    
    for i in range(0, len(updates), _bulk_write_size):
        batch = updates[i:i + _bulk_write_size]
        operations = [UpdateOne(filter_doc, update_doc) for filter_doc, update_doc in batch]
        
        try:
            result = await collection.bulk_write(operations, ordered=ordered)
            total_modified += result.modified_count
        except Exception as e:
            logger.error(f"Bulk update error at batch {i // _bulk_write_size}: {e}")
            if ordered:
                raise
    
    logger.debug(f"Bulk update completed: {total_modified} documents modified")
    return total_modified


def get_bulk_write_size() -> int:
    """Get the optimized bulk write size for external use."""
    return _bulk_write_size

# General collections
collection_case = db["cases"]
users_collection = db["Users"]
alerts_collection = db["Alerts"]
processing_profiles_collection = db["Processing Profiles"]
models_master_collection = db["Models_Master"]
models_repository_collection = db["Models_Repository"]
platform_control_collection = db["Platform Control"]
ufdr_files_collection = db["ufdr_files"]
# UFDR shared collections for all files
ufdr_calls_collection = db["ufdr_calls"]
ufdr_chats_collection = db["ufdr_chats"]
ufdr_emails_collection = db["ufdr_emails"]
ufdr_locations_collection = db["ufdr_locations"]
ufdr_notes_collection = db["ufdr_notes"]
ufdr_searched_items_collection = db["ufdr_searched_items"]
ufdr_user_accounts_collection = db["ufdr_user_accounts"]
ufdr_audio_collection = db["ufdr_audio"]
ufdr_photos_collection = db["ufdr_photos"]
ufdr_videos_collection = db["ufdr_videos"]
ufdr_photo_detected_faces_collection = db["ufdr_photo_detected_faces"]
ufdr_video_detected_faces_collection = db["ufdr_video_detected_faces"]
ufdr_photo_detected_objects_collection = db["ufdr_photo_detected_objects"]
ufdr_video_detected_objects_collection = db["ufdr_video_detected_objects"]
ufdr_video_screenshots_collection = db["ufdr_video_screenshots"]
# Detectors collections
detectors_collection = db["detectors"]
detector_matches_collection = db["detector_matches"]
detector_settings_collection = db["detector_settings"]