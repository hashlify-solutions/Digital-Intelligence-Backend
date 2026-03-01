import os
import datetime
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from config.db import models_repository_collection, db
from bson import json_util
import json
from pathlib import Path
from config.settings import settings
import urllib.request
import shutil
import ssl
import subprocess


def setup_logging():
    """Set up detailed logging."""
    log_dir = settings.log_dir
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get the root logger
    logger = logging.getLogger()

    # Check if the logger already has handlers to avoid duplicates
    if not logger.hasHandlers():
        # File handler
        file_handler = logging.FileHandler(
            f"{log_dir}/logfile_{current_time}.log", encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        console_handler.setLevel(logging.INFO)

        # Configure the logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


async def download_models():
    try:
        models_repository = await models_repository_collection.find().to_list(None)

        print(models_repository)

        if models_repository and len(models_repository) > 0:

            for model in models_repository:
                model_name = model.get("name")
                model_dir = f"./models/{model.get('model')}"
                if model.get("is_local", True):
                    if not os.path.exists(model_dir):
                        logger.info(f"Downloading {model_name} model...")
                        # Load model & tokenizer
                        if (
                            model_name
                            == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                        ):
                            auto_model = AutoModel.from_pretrained(model_name)
                        else:
                            auto_model = (
                                AutoModelForSequenceClassification.from_pretrained(
                                    model_name
                                )
                            )
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        # Save them
                        os.makedirs(model_dir, exist_ok=True)
                        auto_model.save_pretrained(model_dir)
                        tokenizer.save_pretrained(model_dir)
                    else:
                        logger.info(
                            f"Model {model_name} already exists. Skipping download."
                        )
                else:
                    logger.info(f"Model {model_name} is not local. Skipping download.")

        else:
            logger.warning("No models found in the master profile.")
            return

    except Exception as e:
        logger.error(
            f"Something went wrong while downloading models: {e}", exc_info=True
        )
        raise


async def seed_collections():
    """Seed MongoDB collections with data from JSON files if collections are empty."""
    try:
        # Define collection-file mappings
        collection_mappings = {
            "Models_Master": "DigitalIntelligence.Models_Master.json",
            "Models_Repository": "DigitalIntelligence.Models_Repository.json",
        }

        # Check and seed each collection
        for collection_name, json_file in collection_mappings.items():
            # Get the collection
            collection = db[collection_name]

            # Check if collection exists and is empty
            if await collection.count_documents({}) == 0:
                logger.info(f"Seeding {collection_name} collection...")

                # Construct path to JSON file
                json_path = Path("seeders") / json_file

                if not json_path.exists():
                    logger.warning(
                        f"JSON file {json_path} not found. Skipping seeding for {collection_name}."
                    )
                    continue

                # Load and parse JSON data
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Insert data into collection
                if data:
                    # Convert string ObjectIds to proper ObjectId objects
                    parsed_data = json_util.loads(json_util.dumps(data))
                    await collection.insert_many(parsed_data)
                    logger.info(
                        f"Successfully seeded {collection_name} with {len(data)} documents."
                    )
                else:
                    logger.warning(
                        f"No data found in {json_path} for {collection_name}."
                    )
            else:
                logger.info(
                    f"{collection_name} collection already contains data. Skipping seeding."
                )

    except Exception as e:
        logger.error(f"Error seeding collections: {e}", exc_info=True)
        raise


# def setup_directories():
#     """Setup required directories for the application."""
#     try:
#         logger.info("Setting up application directories...")

#         # Normalize all directory paths
#         upload_dir = os.path.normpath(settings.upload_dir)
#         logo_dir = os.path.normpath(settings.logo_dir)
#         log_dir = os.path.normpath(settings.log_dir)

#         # Create required directories
#         directories = [
#             upload_dir,
#             logo_dir,
#             log_dir,
#             "models/dnn-face-detector",
#             "models/yolo12x-object-detector",
#             "models/classifier",
#             "models/embeddings",
#             "models/emotion",
#             "models/entity",
#             "models/MiniLM-L6-v2",
#             "models/toxic",
#         ]

#         for directory in directories:
#             os.makedirs(directory, exist_ok=True)
#             logger.info(f"Created directory: {directory}")

#         logger.info("Directory setup completed successfully.")

#     except Exception as e:
#         logger.error(f"Directory setup failed: {e}", exc_info=True)
#         raise


def download_file(url: str, file_path: str) -> bool:
    """Download a file from URL to the specified path."""
    try:
        # Create a context that doesn't verify SSL certificates
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        # For GitHub raw files that need different handling
        if "raw.githubusercontent.com" in url or "github.com" in url:
            # Use wget for GitHub files as it handles redirects better
            subprocess.run(
                ["wget", "--no-check-certificate", "-O", file_path, url], check=True
            )
        else:
            # For direct downloads, use urllib
            with urllib.request.urlopen(url, context=context) as response:
                with open(file_path, "wb") as out_file:
                    shutil.copyfileobj(response, out_file)
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False


def setup_external_models():
    """Setup external models that need manual download."""
    try:
        logger.info("Checking and downloading external model files...")

        # Define required model files and their paths with direct download URLs
        required_files = {
            "models/dnn-face-detector/deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "models/dnn-face-detector/res10_300x300_ssd_iter_140000.caffemodel": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "models/yolo12x-object-detector/yolo12x.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt",
        }

        # Track download status
        downloaded_files = []
        failed_downloads = []
        existing_files = []

        # Check and download each file
        for file_path, download_url in required_files.items():
            if os.path.exists(file_path):
                existing_files.append(file_path)
                continue

            logger.info(f"Downloading {os.path.basename(file_path)}...")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if download_file(download_url, file_path):
                downloaded_files.append(file_path)
                logger.info(f"Successfully downloaded {os.path.basename(file_path)}")
            else:
                failed_downloads.append((file_path, download_url))

        # Log summary
        if existing_files:
            logger.info("\nExisting files (already downloaded):")
            for file in existing_files:
                logger.info(f"✓ {file}")

        if downloaded_files:
            logger.info("\nNewly downloaded files:")
            for file in downloaded_files:
                logger.info(f"✓ {file}")

        if failed_downloads:
            logger.warning("\nFailed downloads (please download manually):")
            for file_path, url in failed_downloads:
                logger.warning(f"✗ {file_path}")
                logger.warning(f"  URL: {url}")

        if not failed_downloads:
            logger.info("\n✓ All external models are set up successfully!")

    except Exception as e:
        logger.error(f"External model setup failed: {e}", exc_info=True)
        raise


async def setup_database():
    """Setup database collections and seed initial data if needed."""
    try:
        logger.info("Starting database setup...")

        # Ensure collections exist (they'll be created automatically on first insert)
        # Seed collections with initial data if empty
        await seed_collections()

        logger.info("Database setup completed successfully.")

    except Exception as e:
        logger.error(f"Database setup failed: {e}", exc_info=True)
        raise


async def setup_application():
    """Main setup function that orchestrates the entire application setup."""
    try:
        logger.info("Starting Digital Intelligence Platform setup...")

        # Step 1: Setup database and seed data
        await setup_database()

        # Step 2: Download Hugging Face models
        # await download_models()

        # Step 3: Check external models
        setup_external_models()

        logger.info("Digital Intelligence Platform setup completed successfully!")
        logger.info("Please check the logs above for any manual steps required.")

    except Exception as e:
        logger.error(f"Application setup failed: {e}", exc_info=True)
        raise


logger = setup_logging()