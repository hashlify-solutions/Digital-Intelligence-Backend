import os
import logging
import datetime
import zipfile
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
import tempfile
import shutil
import email
from email import policy
from email.parser import BytesParser
from urllib.parse import urlparse
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import required packages
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from config.db import qdrant_client
from config.settings import settings
import numpy as np
from rag import ArabicRagAnalyzer

# Configure logging
logging.basicConfig(
    filename="ufdr_ingestion.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class UFDRIngester:
    def __init__(
        self,
        mongo_collection: AsyncIOMotorCollection,
        mongo_collection_all_cases: AsyncIOMotorCollection,
        case_id: str,
        models_profile: Dict = None,
    ):
        self.mongo_collection = mongo_collection
        self.collection_all_cases = mongo_collection_all_cases
        self.case_id = case_id
        self.models_profile = models_profile
        self.temp_dir = None
        
        # Get configuration from centralized settings
        compute = settings.compute_config
        self.batch_size = compute["batch_size"]
        self.max_workers = compute["max_workers"]
        self.chunk_size = compute["chunk_size"]
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(
            f"UFDRIngester initialized: batch_size={self.batch_size}, "
            f"workers={self.max_workers}, chunk_size={self.chunk_size}"
        )

    def extract_url_metadata(self, url):
        """Extract metadata from URL including domain, path, and query parameters"""
        if not url:
            return {}

        try:
            parsed = urlparse(url)
            metadata = {
                "domain": parsed.netloc,
                "path": parsed.path,
                "query": parsed.query,
                "fragment": parsed.fragment,
                "scheme": parsed.scheme,
            }

            # Extract common domain patterns
            if parsed.netloc:
                parts = parsed.netloc.split(".")
                if len(parts) >= 2:
                    metadata["main_domain"] = ".".join(parts[-2:])
                    metadata["subdomain"] = (
                        ".".join(parts[:-2]) if len(parts) > 2 else None
                    )

            return metadata
        except Exception as e:
            logger.warning(f"Error parsing URL {url}: {e}")
            return {}

    def categorize_browsing_activity(self, url, title):
        """Categorize browsing activity based on URL and title"""
        if not url and not title:
            return "Unknown"

        url_lower = (url or "").lower()
        title_lower = (title or "").lower()

        # Social Media
        if any(
            site in url_lower
            for site in [
                "facebook.com",
                "twitter.com",
                "instagram.com",
                "linkedin.com",
                "youtube.com",
            ]
        ):
            return "Social Media"

        # Search Engines
        if any(
            site in url_lower
            for site in ["google.com/search", "bing.com/search", "yahoo.com/search"]
        ):
            return "Search Engine"

        # Email Services
        if any(
            site in url_lower
            for site in ["gmail.com", "outlook.com", "yahoo.com/mail", "hotmail.com"]
        ):
            return "Email"

        # News and Information
        if any(
            site in url_lower for site in ["news.", "bbc.com", "cnn.com", "reuters.com"]
        ):
            return "News"

        # Shopping/E-commerce
        if any(
            site in url_lower for site in ["amazon.com", "ebay.com", "shop.", "store."]
        ):
            return "Shopping"

        # Banking/Finance
        if any(
            site in url_lower for site in ["bank.", "paypal.com", "finance.", "money."]
        ):
            return "Banking/Finance"

        # Educational
        if any(
            site in url_lower for site in ["edu.", "university.", "college.", "school."]
        ):
            return "Educational"

        # Government
        if any(site in url_lower for site in ["gov.", "government."]):
            return "Government"

        # Entertainment
        if any(
            site in url_lower
            for site in ["netflix.com", "spotify.com", "game.", "entertainment."]
        ):
            return "Entertainment"

        return "General Web"

    def categorize_media_type(self, file_path, tags, file_extension):
        """Categorize media files based on path, tags, and extension."""
        path_lower = (file_path or "").lower()
        tags_lower = (tags or "").lower()
        ext_lower = (file_extension or "").lower()

        # Image files - check both extension and path
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
            ".ico",
            ".svg",
            ".thm",
        ]
        if ext_lower in image_extensions:
            return "image"
        if any(path_lower.endswith(ext) for ext in image_extensions):
            return "image"
        if "image" in tags_lower or "photo" in path_lower or "thumb" in path_lower:
            return "image"

        # Video files
        video_extensions = [
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".mkv",
            ".webm",
            ".3gp",
            ".m4v",
            ".mpg",
            ".mpeg",
        ]
        if ext_lower in video_extensions:
            return "video"
        if any(path_lower.endswith(ext) for ext in video_extensions):
            return "video"
        if "video" in tags_lower or "movie" in path_lower:
            return "video"

        # Audio files
        audio_extensions = [
            ".mp3",
            ".wav",
            ".aac",
            ".m4a",
            ".flac",
            ".ogg",
            ".wma",
            ".mmf",
            ".amr",
            ".3gp",
        ]
        if ext_lower in audio_extensions:
            return "audio"
        if any(path_lower.endswith(ext) for ext in audio_extensions):
            return "audio"
        if "audio" in tags_lower or "sound" in path_lower or "music" in path_lower:
            return "audio"

        # Document files
        document_extensions = [
            ".pdf",
            ".doc",
            ".docx",
            ".txt",
            ".rtf",
            ".odt",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
        ]
        if ext_lower in document_extensions:
            return "document"
        if any(path_lower.endswith(ext) for ext in document_extensions):
            return "document"
        if "document" in tags_lower or "text" in tags_lower:
            return "document"

        # Archive files
        archive_extensions = [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"]
        if ext_lower in archive_extensions:
            return "archive"
        if any(path_lower.endswith(ext) for ext in archive_extensions):
            return "archive"
        if "archive" in tags_lower or "compressed" in tags_lower:
            return "archive"

        return "other"

    def extract_file_metadata(self, file_element, ns):
        """Extract comprehensive metadata from a file element."""
        metadata = {}

        # Basic file attributes
        metadata["file_id"] = file_element.get("id")
        metadata["file_path"] = file_element.get("path")
        metadata["file_size"] = file_element.get("size")
        metadata["file_system"] = file_element.get("fs")
        metadata["file_system_id"] = file_element.get("fsid")
        metadata["extraction_id"] = file_element.get("extractionId")
        metadata["deleted_state"] = file_element.get("deleted", "Intact")
        metadata["embedded"] = file_element.get("embedded", "false")
        metadata["is_related"] = file_element.get("isrelated", "False")

        # Extract file extension
        if metadata["file_path"]:
            metadata["file_extension"] = os.path.splitext(metadata["file_path"])[
                1
            ].lower()
            metadata["file_name"] = os.path.basename(metadata["file_path"])
        else:
            metadata["file_extension"] = ""
            metadata["file_name"] = ""

        # Access info (timestamps)
        access_info = file_element.find("ns:accessInfo" if ns else "accessInfo", ns)
        if access_info is not None:
            metadata["creation_time"] = access_info.findtext(
                (
                    'ns:timestamp[@name="CreationTime"]'
                    if ns
                    else 'timestamp[@name="CreationTime"]'
                ),
                namespaces=ns,
            )
            metadata["modify_time"] = access_info.findtext(
                (
                    'ns:timestamp[@name="ModifyTime"]'
                    if ns
                    else 'timestamp[@name="ModifyTime"]'
                ),
                namespaces=ns,
            )
            metadata["access_time"] = access_info.findtext(
                (
                    'ns:timestamp[@name="AccessTime"]'
                    if ns
                    else 'timestamp[@name="AccessTime"]'
                ),
                namespaces=ns,
            )

        # File metadata section
        file_metadata = file_element.find(
            'ns:metadata[@section="File"]' if ns else 'metadata[@section="File"]', ns
        )
        if file_metadata is not None:
            metadata["local_path"] = file_metadata.findtext(
                'ns:item[@name="Local Path"]' if ns else 'item[@name="Local Path"]',
                namespaces=ns,
            )
            metadata["sha256"] = file_metadata.findtext(
                'ns:item[@name="SHA256"]' if ns else 'item[@name="SHA256"]',
                namespaces=ns,
            )
            metadata["md5"] = file_metadata.findtext(
                'ns:item[@name="MD5"]' if ns else 'item[@name="MD5"]', namespaces=ns
            )
            metadata["tags"] = file_metadata.findtext(
                'ns:item[@name="Tags"]' if ns else 'item[@name="Tags"]', namespaces=ns
            )

        # Additional metadata section
        additional_metadata = file_element.find(
            (
                'ns:metadata[@section="MetaData"]'
                if ns
                else 'metadata[@section="MetaData"]'
            ),
            ns,
        )
        if additional_metadata is not None:
            # Extract all metadata items
            metadata_items = {}
            for item in additional_metadata.findall("ns:item" if ns else "item", ns):
                name = item.get("name")
                group = item.get("group")
                value = item.text
                if name:
                    key = f"{group}_{name}" if group else name
                    metadata_items[key] = value
            metadata["additional_metadata"] = metadata_items

            # Extract specific common fields
            metadata["data_offset"] = additional_metadata.findtext(
                'ns:item[@name="Data offset"]' if ns else 'item[@name="Data offset"]',
                namespaces=ns,
            )
            metadata["file_size_bytes"] = additional_metadata.findtext(
                'ns:item[@name="File size"]' if ns else 'item[@name="File size"]',
                namespaces=ns,
            )
            metadata["chunks"] = additional_metadata.findtext(
                'ns:item[@name="Chunks"]' if ns else 'item[@name="Chunks"]',
                namespaces=ns,
            )

            # EXIF data for images
            exif_data = {}
            for item in additional_metadata.findall(
                'ns:item[@group="EXIF"]' if ns else 'item[@group="EXIF"]', ns
            ):
                name = item.get("name")
                value = item.text
                if name:
                    exif_data[name] = value
            if exif_data:
                metadata["exif_data"] = exif_data

        # Categorize media type
        metadata["media_type"] = self.categorize_media_type(
            metadata["file_path"], metadata["tags"], metadata["file_extension"]
        )

        return metadata

    async def process_media_files(self, root, ns, xml_filename):
        """Process all media files from taggedFiles section."""
        try:
            logger.info("Processing media files from UFDR XML...")

            # Find taggedFiles section
            tagged_files = root.find("ns:taggedFiles" if ns else "taggedFiles", ns)
            if tagged_files is None:
                logger.info("No taggedFiles section found in UFDR XML")
                return

            media_counts = {
                "image": 0,
                "audio": 0,
                "video": 0,
                "document": 0,
                "archive": 0,
                "other": 0,
            }
            media_batch = []

            for file_element in tagged_files.findall("ns:file" if ns else "file", ns):
                try:
                    # Extract comprehensive metadata
                    file_metadata = self.extract_file_metadata(file_element, ns)

                    # Skip if no essential data
                    if not file_metadata.get("file_path") and not file_metadata.get(
                        "local_path"
                    ):
                        continue

                    # Create media record
                    media_record = {
                        "file_id": file_metadata.get("file_id"),
                        "file_name": file_metadata.get("file_name"),
                        "file_path": file_metadata.get("file_path"),
                        "local_path": file_metadata.get("local_path"),
                        "file_size": file_metadata.get("file_size"),
                        "file_size_bytes": file_metadata.get("file_size_bytes"),
                        "file_extension": file_metadata.get("file_extension"),
                        "media_type": file_metadata.get("media_type"),
                        "tags": file_metadata.get("tags"),
                        "sha256": file_metadata.get("sha256"),
                        "md5": file_metadata.get("md5"),
                        "creation_time": file_metadata.get("creation_time"),
                        "modify_time": file_metadata.get("modify_time"),
                        "access_time": file_metadata.get("access_time"),
                        "deleted_state": file_metadata.get("deleted_state"),
                        "embedded": file_metadata.get("embedded"),
                        "is_related": file_metadata.get("is_related"),
                        "file_system": file_metadata.get("file_system"),
                        "file_system_id": file_metadata.get("file_system_id"),
                        "extraction_id": file_metadata.get("extraction_id"),
                        "data_offset": file_metadata.get("data_offset"),
                        "chunks": file_metadata.get("chunks"),
                        "exif_data": file_metadata.get("exif_data", {}),
                        "additional_metadata": file_metadata.get(
                            "additional_metadata", {}
                        ),
                        "case_id": ObjectId(self.case_id),
                        "source_file": xml_filename,
                        "source": "UFDR-Media",
                        "processed": True,
                        "ingestion_timestamp": datetime.datetime.now().isoformat(),
                    }

                    media_batch.append(media_record)

                    # Count by media type
                    media_type = file_metadata.get("media_type", "other")
                    if media_type in media_counts:
                        media_counts[media_type] += 1
                    else:
                        media_counts["other"] += 1

                    # Debug logging for first few files
                    if media_counts["other"] < 10:
                        logger.info(
                            f"XML File: {file_metadata.get('file_name')}, Extension: {file_metadata.get('file_extension')}, Type: {media_type}"
                        )

                    # Flush batch when it reaches threshold
                    if len(media_batch) >= 500:
                        await self.mongo_collection.insert_many(media_batch, ordered=False)
                        logger.info(f"Bulk inserted {len(media_batch)} media records")
                        media_batch = []

                except Exception as e:
                    logger.error(f"Error processing media file: {str(e)}")
                    continue

            # Flush remaining records
            if media_batch:
                await self.mongo_collection.insert_many(media_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(media_batch)} media records")

            logger.info(f"Media files processed: {media_counts}")
            print(f"[Media Extraction] Processed from {xml_filename}: {media_counts}")

        except Exception as e:
            logger.error(f"Error in process_media_files: {str(e)}")
            raise

    def get_file_metadata_from_path(self, file_path, xml_filename):
        """Extract metadata from file path and system."""
        try:
            import os
            import hashlib
            import mimetypes

            # Get file stats
            stat = os.stat(file_path)
            file_size = stat.st_size

            # Get file extension and name
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1].lower()

            # Calculate hashes
            sha256_hash = None
            md5_hash = None
            try:
                with open(file_path, "rb") as f:
                    # Read file in chunks for large files
                    sha256_hasher = hashlib.sha256()
                    md5_hasher = hashlib.md5()
                    while chunk := f.read(8192):
                        sha256_hasher.update(chunk)
                        md5_hasher.update(chunk)
                    sha256_hash = sha256_hasher.hexdigest().upper()
                    md5_hash = md5_hasher.hexdigest()
            except Exception as e:
                logger.warning(f"Could not calculate hashes for {file_path}: {e}")

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(file_path)

            # Determine media type
            media_type = self.categorize_media_type(file_path, "", file_extension)

            # Get relative path from temp directory
            relative_path = os.path.relpath(file_path, self.temp_dir)

            return {
                "file_name": file_name,
                "file_path": relative_path,
                "local_path": relative_path,
                "file_size": str(file_size),
                "file_size_bytes": f"{file_size} Bytes",
                "file_extension": file_extension,
                "media_type": media_type,
                "sha256": sha256_hash,
                "md5": md5_hash,
                "mime_type": mime_type,
                "creation_time": datetime.datetime.fromtimestamp(
                    stat.st_ctime
                ).isoformat(),
                "modify_time": datetime.datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat(),
                "access_time": datetime.datetime.fromtimestamp(
                    stat.st_atime
                ).isoformat(),
                "deleted_state": "Intact",  # Files found in directory are intact
                "embedded": "false",
                "is_related": "True",  # Files in directory are related
                "file_system": "Extracted",
                "source": "UFDR-Directory-Scan",
            }

        except Exception as e:
            logger.error(f"Error getting metadata for {file_path}: {e}")
            return None

    async def scan_directory_for_media_files(self, xml_filename):
        """Scan the entire extracted UFDR directory for media files."""
        try:
            logger.info("Scanning extracted UFDR directory for media files...")

            # Define media file extensions
            media_extensions = {
                "image": [
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".bmp",
                    ".tiff",
                    ".webp",
                    ".ico",
                    ".svg",
                    ".thm",
                ],
                "audio": [
                    ".mp3",
                    ".wav",
                    ".aac",
                    ".m4a",
                    ".flac",
                    ".ogg",
                    ".wma",
                    ".mmf",
                    ".amr",
                    ".3gp",
                ],
                "video": [
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".wmv",
                    ".flv",
                    ".mkv",
                    ".webm",
                    ".3gp",
                    ".m4v",
                    ".mpg",
                    ".mpeg",
                ],
                "document": [
                    ".pdf",
                    ".doc",
                    ".docx",
                    ".txt",
                    ".rtf",
                    ".odt",
                    ".xls",
                    ".xlsx",
                    ".ppt",
                    ".pptx",
                ],
                "archive": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"],
                "other": [
                    ".exe",
                    ".dll",
                    ".sys",
                    ".bin",
                    ".dat",
                    ".log",
                    ".ini",
                    ".cfg",
                    ".xml",
                    ".json",
                    ".plist",
                    ".sqlite",
                    ".db",
                ],
            }

            # Flatten all extensions for checking
            all_media_extensions = []
            for ext_list in media_extensions.values():
                all_media_extensions.extend(ext_list)

            media_counts = {
                "image": 0,
                "audio": 0,
                "video": 0,
                "document": 0,
                "archive": 0,
                "other": 0,
            }
            processed_files = []
            media_batch = []

            # Walk through the entire directory structure
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    # Skip macOS resource fork files (._) and hidden files
                    if file.startswith("._") or file.startswith("."):
                        continue

                    file_path = os.path.join(root, file)
                    file_extension = os.path.splitext(file)[1].lower()

                    # Skip if not a media file
                    if file_extension not in all_media_extensions:
                        continue

                    # Skip XML files (already processed)
                    if file.lower().endswith(".xml"):
                        continue

                    # Skip very large files (>100MB) to avoid memory issues
                    try:
                        file_size = os.path.getsize(file_path)
                        if file_size > 100 * 1024 * 1024:  # 100MB
                            logger.warning(
                                f"Skipping large file: {file_path} ({file_size / (1024*1024):.1f}MB)"
                            )
                            continue
                    except Exception:
                        continue

                    try:
                        # Extract metadata
                        file_metadata = self.get_file_metadata_from_path(
                            file_path, xml_filename
                        )
                        if not file_metadata:
                            continue

                        # Create media record
                        media_record = {
                            "file_id": f"dir_scan_{len(processed_files)}",
                            "file_name": file_metadata.get("file_name"),
                            "file_path": file_metadata.get("file_path"),
                            "local_path": file_metadata.get("local_path"),
                            "file_size": file_metadata.get("file_size"),
                            "file_size_bytes": file_metadata.get("file_size_bytes"),
                            "file_extension": file_metadata.get("file_extension"),
                            "media_type": file_metadata.get("media_type"),
                            "tags": f"Directory-Scan-{file_metadata.get('media_type', 'other').title()}",
                            "sha256": file_metadata.get("sha256"),
                            "md5": file_metadata.get("md5"),
                            "mime_type": file_metadata.get("mime_type"),
                            "creation_time": file_metadata.get("creation_time"),
                            "modify_time": file_metadata.get("modify_time"),
                            "access_time": file_metadata.get("access_time"),
                            "deleted_state": file_metadata.get("deleted_state"),
                            "embedded": file_metadata.get("embedded"),
                            "is_related": file_metadata.get("is_related"),
                            "file_system": file_metadata.get("file_system"),
                            "source_file": xml_filename,
                            "source": file_metadata.get("source"),
                            "case_id": ObjectId(self.case_id),
                            "processed": True,
                            "ingestion_timestamp": datetime.datetime.now().isoformat(),
                            "scan_method": "directory_traversal",
                        }

                        media_batch.append(media_record)
                        processed_files.append(media_record)

                        # Count by media type
                        media_type = file_metadata.get("media_type", "other")
                        if media_type in media_counts:
                            media_counts[media_type] += 1
                        else:
                            media_counts["other"] += 1

                        # Debug logging for first few files
                        if len(processed_files) < 10:
                            logger.info(
                                f"File: {file_metadata.get('file_name')}, Extension: {file_metadata.get('file_extension')}, Type: {media_type}"
                            )

                        # Log progress for large directories
                        if len(processed_files) % 100 == 0:
                            logger.info(
                                f"Processed {len(processed_files)} media files from directory scan..."
                            )

                        # Flush batch when it reaches threshold
                        if len(media_batch) >= 500:
                            await self.mongo_collection.insert_many(media_batch, ordered=False)
                            logger.info(f"Bulk inserted {len(media_batch)} media records from directory scan")
                            media_batch = []

                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {str(e)}")
                        continue

            # Flush remaining records
            if media_batch:
                await self.mongo_collection.insert_many(media_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(media_batch)} media records from directory scan")

            logger.info(f"Directory scan completed. Found media files: {media_counts}")
            print(
                f"[Directory Scan] Found media files from {xml_filename}: {media_counts}"
            )

        except Exception as e:
            logger.error(f"Error in scan_directory_for_media_files: {str(e)}")
            raise

    async def extract_embedded_content_from_xml(self, root, ns, xml_filename):
        """Extract any embedded content (base64, binary data) from XML."""
        try:
            logger.info("Scanning XML for embedded content...")

            embedded_count = 0
            self._embedded_batch = []

            # Look for embedded content in various XML sections
            sections_to_check = [
                "ns:decodedData" if ns else "decodedData",
                "ns:taggedFiles" if ns else "taggedFiles",
                "ns:carvedFiles" if ns else "carvedFiles",
                "ns:infectedFiles" if ns else "infectedFiles",
            ]

            for section_name in sections_to_check:
                section = root.find(section_name, ns)
                if section is not None:
                    # Look for embedded content in models or files
                    for element in section.findall(
                        ".//ns:model" if ns else ".//model", ns
                    ):
                        await self.extract_embedded_from_element(
                            element, ns, xml_filename, embedded_count
                        )
                        embedded_count += 1

                    for element in section.findall(
                        ".//ns:file" if ns else ".//file", ns
                    ):
                        await self.extract_embedded_from_element(
                            element, ns, xml_filename, embedded_count
                        )
                        embedded_count += 1

                    # Flush batch periodically during section processing
                    if len(self._embedded_batch) >= 500:
                        await self.mongo_collection.insert_many(self._embedded_batch, ordered=False)
                        logger.info(f"Bulk inserted {len(self._embedded_batch)} embedded records")
                        self._embedded_batch = []

            # Look for base64 encoded content in any field
            for field in root.findall(".//ns:field" if ns else ".//field", ns):
                await self.extract_base64_from_field(field, ns, xml_filename)
                # Flush batch periodically during field processing
                if len(self._embedded_batch) >= 500:
                    await self.mongo_collection.insert_many(self._embedded_batch, ordered=False)
                    logger.info(f"Bulk inserted {len(self._embedded_batch)} embedded records from field scan")
                    self._embedded_batch = []

            # Flush remaining embedded records
            if self._embedded_batch:
                await self.mongo_collection.insert_many(self._embedded_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(self._embedded_batch)} embedded records")
                self._embedded_batch = []

            logger.info(
                f"Embedded content extraction completed. Processed {embedded_count} elements."
            )

        except Exception as e:
            logger.error(f"Error in extract_embedded_content_from_xml: {str(e)}")
            # Don't raise - this is supplementary extraction

    async def extract_embedded_from_element(self, element, ns, xml_filename, count):
        """Extract embedded content from a specific XML element."""
        try:
            # Look for embedded data in various field names
            embedded_fields = [
                "data",
                "content",
                "payload",
                "attachment",
                "binary",
                "image",
                "video",
                "audio",
            ]

            for field_name in embedded_fields:
                field = element.find(
                    (
                        f'ns:field[@name="{field_name}"]'
                        if ns
                        else f'field[@name="{field_name}"]'
                    ),
                    ns,
                )
                if field is not None:
                    value_elem = field.find("ns:value" if ns else "value", ns)
                    if value_elem is not None and value_elem.text:
                        content = value_elem.text.strip()
                        if self.is_base64_content(content):
                            await self.process_base64_content(
                                content, field_name, element, xml_filename, count
                            )

        except Exception as e:
            logger.debug(f"Error extracting embedded content from element: {e}")

    async def extract_base64_from_field(self, field, ns, xml_filename):
        """Extract base64 content from a field."""
        try:
            value_elem = field.find("ns:value" if ns else "value", ns)
            if value_elem is not None and value_elem.text:
                content = value_elem.text.strip()
                if self.is_base64_content(content):
                    field_name = field.get("name", "unknown")
                    await self.process_base64_content(
                        content, field_name, field, xml_filename, 0
                    )

        except Exception as e:
            logger.debug(f"Error extracting base64 from field: {e}")

    def is_base64_content(self, content):
        """Check if content appears to be base64 encoded."""
        if not content or len(content) < 100:  # Too short to be meaningful media
            return False

        # Check for common base64 patterns
        base64_patterns = [
            "iVBORw0KGgo",  # PNG
            "/9j/",  # JPEG
            "UklGR",  # WebP
            "data:image",  # Data URI
            "data:video",  # Data URI
            "data:audio",  # Data URI
        ]

        content_start = content[:50]  # Check first 50 characters
        return any(pattern in content_start for pattern in base64_patterns)

    async def process_base64_content(
        self, content, field_name, element, xml_filename, count
    ):
        """Process base64 encoded content."""
        try:
            import base64
            import mimetypes

            # Determine content type
            content_type = self.detect_content_type(content)
            if not content_type:
                return

            # Extract base64 data
            if content.startswith("data:"):
                # Data URI format: data:image/png;base64,iVBORw0KGgo...
                header, data = content.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]
            else:
                # Raw base64
                data = content
                mime_type = content_type

            # Decode base64
            try:
                binary_data = base64.b64decode(data)
            except Exception:
                return  # Invalid base64

            # Generate filename
            extension = mimetypes.guess_extension(mime_type) or ".bin"
            filename = f"embedded_{field_name}_{count}{extension}"

            # Create media record
            media_record = {
                "file_id": f"embedded_{count}",
                "file_name": filename,
                "file_path": f"embedded/{filename}",
                "local_path": f"embedded/{filename}",
                "file_size": str(len(binary_data)),
                "file_size_bytes": f"{len(binary_data)} Bytes",
                "file_extension": extension,
                "media_type": self.categorize_by_mime_type(mime_type),
                "tags": f"Embedded-{field_name}",
                "sha256": self.calculate_sha256(binary_data),
                "md5": self.calculate_md5(binary_data),
                "mime_type": mime_type,
                "creation_time": datetime.datetime.now().isoformat(),
                "modify_time": datetime.datetime.now().isoformat(),
                "access_time": datetime.datetime.now().isoformat(),
                "deleted_state": "Intact",
                "embedded": "true",
                "is_related": "True",
                "file_system": "XML-Embedded",
                "source_file": xml_filename,
                "source": "UFDR-XML-Embedded",
                "case_id": ObjectId(self.case_id),
                "processed": True,
                "ingestion_timestamp": datetime.datetime.now().isoformat(),
                "scan_method": "xml_embedded_extraction",
                "embedded_content": content[
                    :1000
                ],  # Store first 1000 chars for reference
            }

            # Append to batch for bulk insertion (caller handles flushing)
            if hasattr(self, '_embedded_batch'):
                self._embedded_batch.append(media_record)
            else:
                # Fallback to single insert if called outside batch context
                await self.mongo_collection.insert_one(media_record)

        except Exception as e:
            logger.error(f"Error processing base64 content: {e}")

    def detect_content_type(self, content):
        """Detect content type from base64 content."""
        content_start = content[:50]

        if content.startswith("data:"):
            return content.split(":")[1].split(";")[0]
        elif "iVBORw0KGgo" in content_start:
            return "image/png"
        elif "/9j/" in content_start:
            return "image/jpeg"
        elif "UklGR" in content_start:
            return "image/webp"
        elif content.startswith("GIF"):
            return "image/gif"
        else:
            return None

    def categorize_by_mime_type(self, mime_type):
        """Categorize media type by MIME type."""
        if mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("video/"):
            return "video"
        elif mime_type.startswith("audio/"):
            return "audio"
        elif mime_type.startswith("text/"):
            return "document"
        else:
            return "other"

    def calculate_sha256(self, data):
        """Calculate SHA256 hash of binary data."""
        import hashlib

        return hashlib.sha256(data).hexdigest().upper()

    def calculate_md5(self, data):
        """Calculate MD5 hash of binary data."""
        import hashlib

        return hashlib.md5(data).hexdigest()

    async def process_main_xml(self):
        """Scan and extract data from the main XML file in the extracted UFDR directory."""
        try:
            # Find the main XML file (first .xml found)
            main_xml_path = None
            for root_dir, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    if file.lower().endswith(".xml"):
                        main_xml_path = os.path.join(root_dir, file)
                        break
                if main_xml_path:
                    break
            if not main_xml_path:
                logger.info("No main XML file found in extracted UFDR.")
                return

            xml_filename = os.path.basename(main_xml_path)
            logger.info(f"Processing main XML file: {main_xml_path}")
            tree = ET.parse(main_xml_path)
            root = tree.getroot()

            # Namespace handling
            ns = {"ns": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}

            # Process media files first
            await self.process_media_files(root, ns, xml_filename)

            # Also scan the extracted directory structure for media files
            await self.scan_directory_for_media_files(xml_filename)

            # Extract any embedded content from XML
            await self.extract_embedded_content_from_xml(root, ns, xml_filename)

            def get_field(model, name_list):
                # Try flat fields
                for field in model.findall("ns:field" if ns else "field", ns):
                    fname = field.get("name", "").lower()
                    if fname in name_list:
                        value = (
                            field.findtext("ns:value" if ns else "value", namespaces=ns)
                            if ns
                            else field.findtext("value")
                        )
                        if value:
                            return value
                # Try nested modelField
                for fname in name_list:
                    mf = model.find(
                        (
                            f"ns:modelField[@name='{fname.capitalize()}']/ns:model/ns:field[@name='Identifier']/ns:value"
                            if ns
                            else f"modelField[@name='{fname.capitalize()}']/model/field[@name='Identifier']/value"
                        ),
                        ns,
                    )
                    if mf is not None and mf.text:
                        return mf.text.strip()
                return None

            def get_multi_field(model, field_name):
                # For multiModelField (e.g., To, Parties)
                values = []
                for m in model.findall(
                    (
                        f"ns:multiModelField[@name='{field_name}']/ns:model"
                        if ns
                        else f"multiModelField[@name='{field_name}']/model"
                    ),
                    ns,
                ):
                    val = m.find(
                        (
                            "ns:field[@name='Identifier']/ns:value"
                            if ns
                            else "field[@name='Identifier']/value"
                        ),
                        ns,
                    )
                    if val is not None and val.text:
                        values.append(val.text.strip())
                return values

            def extract_sms_parties(model, ns):
                sender = None
                receiver = []
                for party in model.findall(
                    (
                        'ns:multiModelField[@name="Parties"]/ns:model'
                        if ns
                        else 'multiModelField[@name="Parties"]/model'
                    ),
                    ns,
                ):
                    role = None
                    identifier = None
                    for field in party.findall("ns:field" if ns else "field", ns):
                        fname = field.get("name", "").lower()
                        value = (
                            field.findtext("ns:value" if ns else "value", namespaces=ns)
                            if ns
                            else field.findtext("value")
                        )
                        if fname == "role" and value:
                            role = value.strip().lower()
                        if fname == "identifier" and value:
                            identifier = value.strip()
                    if role == "from" and identifier:
                        sender = identifier
                    elif role == "to" and identifier:
                        receiver.append(identifier)
                return sender, receiver

            # Counters for each type
            inserted_counts = {
                "messages": 0,
                "calls": 0,
                "contacts": 0,
                "emails": 0,
                "browsing_history": 0,
                "chats": 0,
                "user_accounts": 0,
                "locations": 0,
                "media_files": 0,
            }
            records_batch = []
            BATCH_FLUSH_SIZE = 500

            # Traverse all modelType/model for robust extraction
            decoded_data = root.find("ns:decodedData" if ns else "decodedData", ns)
            if decoded_data is not None:
                for model_type in decoded_data.findall(
                    "ns:modelType" if ns else "modelType", ns
                ):
                    for model in model_type.findall("ns:model" if ns else "model", ns):
                        try:
                            model_type_val = model.get("type")
                            
                            # --- InstantMessage ---
                            if model_type_val == "InstantMessage":
                                content = get_field(
                                    model, ["body", "text", "content", "message"]
                                )
                                sender = get_field(model, ["from", "sender"])
                                receiver = get_multi_field(model, "To")
                                if not receiver:
                                    val = get_field(
                                        model,
                                        ["to", "recipient", "dest", "destination"],
                                    )
                                    if val:
                                        receiver = [val]
                                timestamp = get_field(
                                    model, ["timestamp", "date", "time"]
                                )
                                receiver_str = "; ".join(receiver) if receiver else None
                                record = {
                                    "Preview Text": content,
                                    "From": sender,
                                    "To": receiver_str,
                                    "Date": timestamp,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "message",
                                    "Application": "SMS/MMS",
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["messages"] += 1
                            
                            # --- SMS ---
                            elif model_type_val == "SMS":
                                content = get_field(
                                    model, ["body", "text", "content", "message"]
                                )
                                timestamp = get_field(
                                    model, ["timestamp", "date", "time", "TimeStamp"]
                                )
                                status = get_field(model, ["status"])
                                folder = get_field(model, ["folder"])
                                sender, receiver_list = extract_sms_parties(model, ns)
                                receiver_str = (
                                    "; ".join(receiver_list) if receiver_list else None
                                )
                                record = {
                                    "Preview Text": content,
                                    "From": sender,
                                    "To": receiver_str,
                                    "Date": timestamp,
                                    "Status": status,
                                    "Folder": folder,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "message",
                                    "Application": "SMS/MMS",
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["messages"] += 1
                            
                            # --- UserAccount ---
                            elif model_type_val == "UserAccount":
                                name = get_field(model, ["name"])
                                username = get_field(model, ["username"])
                                password = get_field(model, ["password"])
                                service_type = get_field(model, ["servicetype"])
                                server_address = get_field(model, ["serveraddress"])
                                record = {
                                    "name": name,
                                    "username": username,
                                    "password": password,
                                    "service_type": service_type,
                                    "server_address": server_address,
                                    "type": "user_account",
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["user_accounts"] += 1
                            
                            # --- Call ---
                            elif model_type_val == "Call":
                                call_type = get_field(model, ["type"])
                                timestamp = get_field(
                                    model, ["timestamp", "date", "time"]
                                )
                                duration = get_field(model, ["duration"])
                                parties = []
                                from_party = None
                                to_parties = []
                                for party_model in model.findall(
                                    (
                                        "ns:multiModelField[@name='Parties']/ns:model"
                                        if ns
                                        else "multiModelField[@name='Parties']/model"
                                    ),
                                    ns,
                                ):
                                    identifier = get_field(party_model, ["identifier"])
                                    role = get_field(party_model, ["role"])
                                    parties.append(
                                        {"identifier": identifier, "role": role}
                                    )
                                    if role and identifier:
                                        if role.lower() == "from":
                                            from_party = identifier
                                        elif role.lower() == "to":
                                            to_parties.append(identifier)
                                to_str = "; ".join(to_parties) if to_parties else None
                                preview = f"{call_type or ''} call, duration: {duration or ''}".strip(
                                    ", "
                                )
                                record = {
                                    "Preview Text": None,
                                    "Name": preview,
                                    "From": from_party,
                                    "To": to_str,
                                    "Date": timestamp,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "call",
                                    "Duration": duration,
                                    "Parties": parties,
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": True,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["calls"] += 1
                            
                            # --- Contact ---
                            elif model_type_val == "Contact":
                                name = get_field(model, ["name"])

                                # Extract phone numbers from nested structure
                                phone_numbers = []
                                entries_field = model.find(
                                    (
                                        "ns:multiModelField[@name='Entries']"
                                        if ns
                                        else "multiModelField[@name='Entries']"
                                    ),
                                    ns,
                                )
                                if entries_field is not None:
                                    for entry_model in entries_field.findall(
                                        "ns:model" if ns else "model", ns
                                    ):
                                        if entry_model.get("type") == "PhoneNumber":
                                            phone_value = get_field(
                                                entry_model, ["value"]
                                            )
                                            phone_category = get_field(
                                                entry_model, ["category"]
                                            )
                                            if phone_value:
                                                phone_info = phone_value
                                                if phone_category:
                                                    phone_info = f"{phone_category}: {phone_value}"
                                                phone_numbers.append(phone_info)

                                # Fallback to flat field if no nested phone numbers found
                                if not phone_numbers:
                                    fallback_phone = get_field(
                                        model, ["phone", "phonenumber"]
                                    )
                                    if fallback_phone:
                                        phone_numbers.append(fallback_phone)

                                # Extract email addresses (similar nested structure might exist)
                                email_addr = get_field(model, ["email"])

                                phone_str = (
                                    "; ".join(phone_numbers) if phone_numbers else None
                                )
                                preview = name or phone_str or email_addr
                                record = {
                                    "Preview Text": (
                                        "; ".join(filter(None, [phone_str, email_addr]))
                                        if phone_str or email_addr
                                        else None
                                    ),
                                    "Name": name,
                                    "From": name,
                                    "To": None,
                                    "Date": None,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "contact",
                                    "Application": "Contact",
                                    "phone_number": phone_str,
                                    "phone_numbers": phone_numbers,  # Store individual phone numbers as array
                                    "email": email_addr,
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": True,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["contacts"] += 1
                           
                            # --- Email ---
                            elif model_type_val == "Email":
                                subject = get_field(model, ["subject"])
                                # Extract sender from <modelField name="From">
                                from_addr = None
                                from_field = model.find(
                                    (
                                        "ns:modelField[@name='From']"
                                        if ns
                                        else "modelField[@name='From']"
                                    ),
                                    ns,
                                )
                                if from_field is not None:
                                    from_model = from_field.find(
                                        "ns:model" if ns else "model", ns
                                    )
                                    if from_model is not None:
                                        from_addr = get_field(
                                            from_model, ["identifier"]
                                        )
                                if not from_addr:
                                    from_addr = get_field(model, ["from", "sender"])

                                # Extract all recipients from <multiModelField name="To">
                                to_addr_list = []
                                to_field = model.find(
                                    (
                                        "ns:multiModelField[@name='To']"
                                        if ns
                                        else "multiModelField[@name='To']"
                                    ),
                                    ns,
                                )
                                if to_field is not None:
                                    for to_model in to_field.findall(
                                        "ns:model" if ns else "model", ns
                                    ):
                                        identifier = get_field(to_model, ["identifier"])
                                        if identifier:
                                            to_addr_list.append(identifier)
                                # Fallback to flat field if not found
                                if not to_addr_list:
                                    fallback_to = get_field(model, ["to", "recipient"])
                                    if fallback_to:
                                        to_addr_list.append(fallback_to)
                                to_addr = (
                                    "; ".join(to_addr_list) if to_addr_list else None
                                )

                                date = get_field(
                                    model, ["date", "timestamp", "TimeStamp"]
                                )
                                body = get_field(
                                    model, ["body", "text", "content", "message"]
                                )
                                record = {
                                    "Preview Text": body,
                                    "From": from_addr,
                                    "To": to_addr,
                                    "Date": date,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "email",
                                    "Application": "Email",
                                    "Subject": subject,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["emails"] += 1
                            
                            # --- SearchedItem ---
                            elif model_type_val == "SearchedItem":
                                search_value = get_field(model, ["value"])
                                timestamp = get_field(model, ["timestamp", "TimeStamp"])
                                source = get_field(model, ["source"])
                                deleted_state = model.get("deleted_state", "Unknown")

                                # Extract search metadata and categorize activity
                                search_metadata = {
                                    "search_engine": source,
                                    "search_type": (
                                        "location"
                                        if any(
                                            word in search_value.lower()
                                            for word in [
                                                "address",
                                                "street",
                                                "road",
                                                "avenue",
                                                "drive",
                                                "lane",
                                                "place",
                                                "bend",
                                                "oregon",
                                            ]
                                        )
                                        else "general"
                                    ),
                                    "search_terms": search_value,
                                }

                                # Categorize search activity
                                if source and "maps" in source.lower():
                                    activity_category = "Maps/Location"
                                elif source and "google" in source.lower():
                                    activity_category = "Search Engine"
                                else:
                                    activity_category = "Search"

                                preview = (
                                    f"Search: {search_value}"
                                    if search_value
                                    else "Search query"
                                )
                                record = {
                                    "Preview Text": preview,
                                    "From": None,
                                    "To": None,
                                    "Date": timestamp,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "browsing_history",
                                    "Application": "Search",
                                    "search_value": search_value,
                                    "search_source": source,
                                    "deleted_state": deleted_state,
                                    "search_id": model.get("id"),
                                    "activity_category": activity_category,
                                    "search_metadata": search_metadata,
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["browsing_history"] += 1
                            
                            # --- WebBookmark ---
                            elif model_type_val == "WebBookmark":
                                title = get_field(model, ["title"])
                                url = get_field(model, ["url"])
                                path = get_field(model, ["path"])
                                timestamp = get_field(model, ["timestamp", "TimeStamp"])
                                deleted_state = model.get("deleted_state", "Unknown")

                                # Extract URL metadata and categorize activity
                                url_metadata = self.extract_url_metadata(url)
                                activity_category = self.categorize_browsing_activity(
                                    url, title
                                )

                                preview = title + " " + url or "Bookmark"
                                record = {
                                    "Preview Text": preview,
                                    "From": None,
                                    "To": None,
                                    "Date": timestamp,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "browsing_history",
                                    "Application": "WebBookmark",
                                    "url": url,
                                    "title": title,
                                    "path": path,
                                    "deleted_state": deleted_state,
                                    "bookmark_id": model.get("id"),
                                    "activity_category": activity_category,
                                    "url_metadata": url_metadata,
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["browsing_history"] += 1
                                
                            # --- VisitedPage ---
                            elif model_type_val == "VisitedPage":
                                title = get_field(model, ["title"])
                                url = get_field(model, ["url"])
                                last_visited = get_field(
                                    model, ["lastvisited", "LastVisited"]
                                )
                                visit_count = get_field(
                                    model, ["visitcount", "VisitCount"]
                                )
                                source = get_field(model, ["source"])
                                deleted_state = model.get("deleted_state", "Unknown")
                                decoding_confidence = model.get(
                                    "decoding_confidence", "Unknown"
                                )
                                is_related = model.get("isrelated", "Unknown")
                                extraction_id = model.get("extractionId", "Unknown")

                                # Extract jumptargets if present
                                jumptargets = []
                                jumptargets_elem = model.find(
                                    "ns:jumptargets" if ns else "jumptargets", ns
                                )
                                if jumptargets_elem is not None:
                                    for target in jumptargets_elem.findall(
                                        "ns:targetid" if ns else "targetid", ns
                                    ):
                                        target_id = (
                                            target.text.strip() if target.text else None
                                        )
                                        is_model = target.get("ismodel", "false")
                                        if target_id:
                                            jumptargets.append(
                                                {
                                                    "target_id": target_id,
                                                    "is_model": is_model.lower()
                                                    == "true",
                                                }
                                            )

                                # Extract URL metadata and categorize activity
                                url_metadata = self.extract_url_metadata(url)
                                activity_category = self.categorize_browsing_activity(
                                    url, title
                                )

                                preview = title + " " + url or "Visited Page"
                                record = {
                                    "Preview Text": preview,
                                    "From": None,
                                    "To": None,
                                    "Date": last_visited,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "browsing_history",
                                    "Application": "VisitedPage",
                                    "url": url,
                                    "title": title,
                                    "last_visited": last_visited,
                                    "visit_count": visit_count,
                                    "browser_source": source,
                                    "deleted_state": deleted_state,
                                    "decoding_confidence": decoding_confidence,
                                    "is_related": is_related,
                                    "extraction_id": extraction_id,
                                    "jumptargets": jumptargets,
                                    "page_id": model.get("id"),
                                    "activity_category": activity_category,
                                    "url_metadata": url_metadata,
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["browsing_history"] += 1
                                
                            # --- Legacy BrowserHistory (fallback) ---
                            elif model_type_val == "BrowserHistory":
                                url = get_field(model, ["url"])
                                title = get_field(model, ["title"])
                                visit_time = get_field(
                                    model, ["timestamp", "date", "time"]
                                )

                                # Extract URL metadata and categorize activity
                                url_metadata = self.extract_url_metadata(url)
                                activity_category = self.categorize_browsing_activity(
                                    url, title
                                )

                                preview = title + " " + url
                                record = {
                                    "Preview Text": preview,
                                    "From": None,
                                    "To": None,
                                    "Date": visit_time,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "browsing_history",
                                    "Application": "BrowserHistory",
                                    "url": url,
                                    "title": title,
                                    "activity_category": activity_category,
                                    "url_metadata": url_metadata,
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["browsing_history"] += 1
                                
                            # --- Chat ---
                            elif model_type_val == "Chat":
                                # Extract chat session metadata if needed
                                chat_participants = []
                                participants_field = model.find(
                                    (
                                        "ns:multiModelField[@name='Participants']"
                                        if ns
                                        else "multiModelField[@name='Participants']"
                                    ),
                                    ns,
                                )
                                if participants_field is not None:
                                    for party_model in participants_field.findall(
                                        "ns:model" if ns else "model", ns
                                    ):
                                        identifier = get_field(
                                            party_model, ["identifier"]
                                        )
                                        if identifier:
                                            chat_participants.append(identifier)
                                chat_participants_str = (
                                    "; ".join(chat_participants)
                                    if chat_participants
                                    else None
                                )

                                # Extract all messages in this chat session
                                messages_field = model.find(
                                    (
                                        "ns:multiModelField[@name='Messages']"
                                        if ns
                                        else "multiModelField[@name='Messages']"
                                    ),
                                    ns,
                                )
                                if messages_field is not None:
                                    for im_model in messages_field.findall(
                                        "ns:model" if ns else "model", ns
                                    ):
                                        # Sender
                                        sender = None
                                        from_field = im_model.find(
                                            (
                                                "ns:modelField[@name='From']"
                                                if ns
                                                else "modelField[@name='From']"
                                            ),
                                            ns,
                                        )
                                        if from_field is not None:
                                            from_model = from_field.find(
                                                "ns:model" if ns else "model", ns
                                            )
                                            if from_model is not None:
                                                sender = get_field(
                                                    from_model, ["identifier"]
                                                )
                                        # Recipients
                                        recipients = []
                                        to_field = im_model.find(
                                            (
                                                "ns:multiModelField[@name='To']"
                                                if ns
                                                else "multiModelField[@name='To']"
                                            ),
                                            ns,
                                        )
                                        if to_field is not None:
                                            for to_model in to_field.findall(
                                                "ns:model" if ns else "model", ns
                                            ):
                                                identifier = get_field(
                                                    to_model, ["identifier"]
                                                )
                                                if identifier:
                                                    recipients.append(identifier)
                                        recipients_str = (
                                            "; ".join(recipients)
                                            if recipients
                                            else None
                                        )
                                        # Body
                                        body = None
                                        body_field = im_model.find(
                                            (
                                                "ns:field[@name='Body']"
                                                if ns
                                                else "field[@name='Body']"
                                            ),
                                            ns,
                                        )
                                        if body_field is not None:
                                            value = body_field.find(
                                                "ns:value" if ns else "value", ns
                                            )
                                            if value is not None and value.text:
                                                body = value.text.strip()
                                        # Timestamp
                                        timestamp = None
                                        ts_field = im_model.find(
                                            (
                                                "ns:field[@name='TimeStamp']"
                                                if ns
                                                else "field[@name='TimeStamp']"
                                            ),
                                            ns,
                                        )
                                        if ts_field is not None:
                                            value = ts_field.find(
                                                "ns:value" if ns else "value", ns
                                            )
                                            if value is not None and value.text:
                                                timestamp = value.text.strip()
                                        # Insert record
                                        record = {
                                            "Preview Text": body,
                                            "From": sender,
                                            "To": recipients_str,
                                            "Date": timestamp,
                                            "Case Name": xml_filename,
                                            "Source": "UFDR-XML",
                                            "Message Type": "chat",
                                            "Application": "Chat",
                                            "Participants": chat_participants_str,
                                            "source_file": xml_filename,
                                            "case_id": ObjectId(self.case_id),
                                            "processed": False,
                                            "alert": None,
                                            "analysis_summary": {
                                                "top_topic": None,
                                                "sentiment_aspect": None,
                                                "emotion": None,
                                                "toxicity_score": None,
                                                "risk_level": None,
                                                "language": None,
                                                "interaction_type": None,
                                                "entities": [],
                                                "entities_classification": {},
                                            },
                                            "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                        }
                                        records_batch.append(record)
                                        inserted_counts["chats"] += 1
                                        
                            # --- Location ---
                            elif model_type_val == "Location":
                                # Extract position coordinates
                                longitude = None
                                latitude = None
                                elevation = None

                                position_field = model.find(
                                    (
                                        "ns:modelField[@name='Position']"
                                        if ns
                                        else "modelField[@name='Position']"
                                    ),
                                    ns,
                                )
                                if position_field is not None:
                                    coordinate_model = position_field.find(
                                        "ns:model" if ns else "model", ns
                                    )
                                    if coordinate_model is not None:
                                        longitude = get_field(
                                            coordinate_model, ["longitude"]
                                        )
                                        latitude = get_field(
                                            coordinate_model, ["latitude"]
                                        )
                                        elevation = get_field(
                                            coordinate_model, ["elevation"]
                                        )

                                # Extract other location fields
                                timestamp = get_field(model, ["timestamp", "TimeStamp"])
                                name = get_field(model, ["name"])
                                description = get_field(model, ["description"])
                                location_type = get_field(model, ["type"])
                                precision = get_field(model, ["precision"])
                                confidence = get_field(model, ["confidence"])
                                category = get_field(model, ["category"])

                                # Create preview text
                                preview_parts = []
                                if name:
                                    preview_parts.append(f"Name: {name}")

                                if category:
                                    preview_parts.append(f"Category: {category}")

                                preview = (
                                    " | ".join(preview_parts)
                                    if preview_parts
                                    else "Location data"
                                )

                                record = {
                                    "Preview Text": preview,
                                    "From": None,
                                    "To": None,
                                    "Date": timestamp,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "location",
                                    "Application": "Location Services",
                                    "latitude": latitude,
                                    "longitude": longitude,
                                    "elevation": elevation,
                                    "description": description,
                                    "location_type": location_type,
                                    "precision": precision,
                                    "confidence": confidence,
                                    "category": category,
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": True,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts["locations"] += 1
                                
                            # --- Note ---
                            elif model_type_val == "Note":
                                title = get_field(model, ["title"])
                                body = get_field(model, ["body"])
                                summary = get_field(model, ["summary"])
                                creation = get_field(model, ["creation"])
                                modification = get_field(model, ["modification"])
                                deleted_state = model.get("deleted_state", "Unknown")
                                note_id = model.get("id")

                                preview = title or body or summary or "Note"
                                record = {
                                    "Preview Text": preview,
                                    "Title": title,
                                    "Body": body,
                                    "Summary": summary,
                                    "Creation": creation,
                                    "Modification": modification,
                                    "deleted_state": deleted_state,
                                    "note_id": note_id,
                                    "Case Name": xml_filename,
                                    "Source": "UFDR-XML",
                                    "Message Type": "note",
                                    "Application": "Note",
                                    "source_file": xml_filename,
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }
                                records_batch.append(record)
                                inserted_counts.setdefault("notes", 0)
                                inserted_counts["notes"] += 1

                            # Flush batch periodically
                            if len(records_batch) >= BATCH_FLUSH_SIZE:
                                await self.mongo_collection.insert_many(records_batch, ordered=False)
                                logger.info(f"Bulk inserted {len(records_batch)} XML model records")
                                records_batch = []
                        except Exception as e:
                            logger.error(f"Error processing model in XML: {str(e)}")

            # Flush remaining records from model traversal
            if records_batch:
                await self.mongo_collection.insert_many(records_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(records_batch)} XML model records")
                records_batch = []

            logger.info(
                f"Finished processing main XML file. Inserted: "
                f"messages={inserted_counts['messages']}, "
                f"calls={inserted_counts['calls']}, "
                f"contacts={inserted_counts['contacts']}, "
                f"emails={inserted_counts['emails']}, "
                f"browsing_history={inserted_counts['browsing_history']} (including SearchedItem, VisitedPage, WebBookmark), "
                f"chats={inserted_counts['chats']}, "
                f"locations={inserted_counts['locations']}, "
                f"notes={inserted_counts.get('notes', 0)}, "
                f"media_files_from_xml={inserted_counts['media_files']} from {xml_filename}"
            )
            print(
                f"[XML Extraction] Inserted from {xml_filename}: "
                f"messages={inserted_counts['messages']}, "
                f"calls={inserted_counts['calls']}, "
                f"contacts={inserted_counts['contacts']}, "
                f"emails={inserted_counts['emails']}, "
                f"browsing_history={inserted_counts['browsing_history']} (including SearchedItem, VisitedPage, WebBookmark), "
                f"chats={inserted_counts['chats']}, "
                f"locations={inserted_counts['locations']}, "
                f"notes={inserted_counts.get('notes', 0)}, "
                f"media_files_from_xml={inserted_counts['media_files']}"
            )
        except Exception as e:
            logger.error(f"Error in process_main_xml: {str(e)}")
            raise

    async def ingest_ufdr_file(self, file_path: str):
        """Process and ingest a UFDR file into MongoDB."""
        try:
            logger.info("Starting UFDR file ingestion...")

            # Update case status
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)},
                {
                    "$set": {
                        "status": "injestion_started",
                        "injesting_started_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    }
                },
            )

            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp()
            logger.info(f"Created temp directory: {self.temp_dir}")

            # Extract UFDR contents
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(self.temp_dir)

            # Process main XML file first (must complete before other file processing)
            await self.process_main_xml()

            # Process data from the extracted files in parallel (independent of each other)
            await asyncio.gather(
                self.process_messages(),
                self.process_chats(),
                self.process_contacts(),
                self.process_emails(),
            )

            # Process RAG data for all extracted data
            await self.ingest_rag_data()

            # Update case status
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)},
                {
                    "$set": {
                        "status": "injestion_completed",
                        "injesting_completed_at": datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat(),
                    }
                },
            )

            logger.info("UFDR ingestion completed successfully")

        except Exception as e:
            logger.error(f"Error during UFDR ingestion: {str(e)}")
            # Update case status to failed
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)},
                {"$set": {"status": "injestion_failed", "error": str(e)}},
            )
            raise
        finally:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

    def find_data_files(self, pattern: str) -> List[str]:
        """Find data files in the extracted UFDR contents."""
        search_patterns = {
            "messages": ["message", "sms", "mms", "im", "chat"],
            "chat": [
                "chat",
                "conversation",
                "im",
                "whatsapp",
                "telegram",
                "signal",
                "imessage",
                "facebook",
            ],
            "contact": ["contact", "phonebook", "address"],
            "email": ["email", "mail", "outlook", "inbox", "sent", "draft"],
        }

        result = []
        patterns_to_search = search_patterns.get(pattern, [pattern])
        allowed_extensions = {
            "messages": (".xml", ".json", ".txt", ".csv", ".db", ".sqlite"),
            "chat": (".xml", ".json", ".txt", ".csv", ".db", ".sqlite"),
            "contact": (".xml", ".json", ".txt", ".csv", ".db", ".sqlite"),
            "email": (".xml", ".json", ".txt", ".csv", ".db", ".sqlite", ".eml"),
        }

        extensions = allowed_extensions.get(
            pattern, (".xml", ".json", ".txt", ".csv", ".db", ".sqlite", ".eml")
        )
        logger.info(
            f"Searching for {pattern} files with patterns: {patterns_to_search}"
        )

        for root, dirs, files in os.walk(self.temp_dir):
            # Check if the directory path contains chat-related words
            dir_path = root.lower()
            is_chat_dir = pattern == "chat" and any(
                p.lower() in dir_path for p in patterns_to_search
            )

            for file in files:
                file_lower = file.lower()
                full_path = os.path.join(root, file)

                # For chat type, include all .txt files in chat-related directories
                if is_chat_dir and file_lower.endswith(".txt"):
                    logger.info(f"Found chat file in chat directory: {full_path}")
                    result.append(full_path)
                    continue

                # For emails, always include .eml files regardless of name
                if pattern == "email" and file_lower.endswith(".eml"):
                    logger.info(f"Found email file: {full_path}")
                    result.append(full_path)
                    continue

                if any(p.lower() in file_lower for p in patterns_to_search):
                    if file_lower.endswith(extensions):
                        logger.info(f"Found matching file: {full_path}")
                        result.append(full_path)

                # Also check parent directory names for relevant patterns
                dir_path = os.path.basename(root).lower()
                if any(p.lower() in dir_path for p in patterns_to_search):
                    if file_lower.endswith(extensions):
                        logger.info(f"Found file in matching directory: {full_path}")
                        result.append(full_path)

        return list(set(result))  # Remove duplicates

    async def process_messages(self):
        """Extract and store messages from the UFDR file."""
        try:
            message_files = self.find_data_files("messages")
            inserted_count = 0
            messages_batch = []
            BATCH_FLUSH_SIZE = 500

            # Update total messages count for messages
            current_total = await self.collection_all_cases.find_one(
                {"_id": ObjectId(self.case_id)}, {"total_messages": 1}
            )
            current_count = (
                current_total.get("total_messages", 0) if current_total else 0
            )

            for file_path in message_files:
                try:
                    # Try different formats - XML, JSON, etc.
                    if file_path.endswith(".xml"):
                        tree = ET.parse(file_path)
                        root = tree.getroot()

                        # Try different XML structures
                        message_elements = (
                            root.findall(".//message")
                            or root.findall(".//sms")
                            or root.findall(".//mms")
                            or root.findall(".//chat")
                            or root.findall(".//instant_message")
                        )

                        for message in message_elements:
                            try:
                                # Try various possible field names
                                content = (
                                    message.findtext("body")
                                    or message.findtext("text")
                                    or message.findtext("content")
                                    or message.findtext("message")
                                )

                                timestamp = (
                                    message.findtext("timestamp")
                                    or message.findtext("date")
                                    or message.findtext("time")
                                )

                                sender = (
                                    message.findtext("from")
                                    or message.findtext("sender")
                                    or message.findtext("source")
                                    or message.findtext("address")
                                )

                                receiver = (
                                    message.findtext("to")
                                    or message.findtext("recipient")
                                    or message.findtext("dest")
                                    or message.findtext("destination")
                                )

                                # Create record matching CSV format
                                record = {
                                    "Preview Text": content,  # Key field for analytics
                                    "From": sender,
                                    "To": receiver,
                                    "Date": timestamp,
                                    "Case Name": os.path.basename(
                                        os.path.dirname(file_path)
                                    ),
                                    "Source": "UFDR",
                                    "Message Type": "message",
                                    "Application": "SMS/MMS",  # Required for analytics
                                    "source_file": os.path.basename(file_path),
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {  # Initialize empty analytics fields
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }

                                if record[
                                    "Preview Text"
                                ]:  # Only insert if there's content
                                    messages_batch.append(record)
                                    inserted_count += 1
                                    if len(messages_batch) >= BATCH_FLUSH_SIZE:
                                        await self.mongo_collection.insert_many(messages_batch, ordered=False)
                                        logger.info(f"Bulk inserted {len(messages_batch)} messages")
                                        messages_batch = []

                            except Exception as e:
                                logger.error(
                                    f"Error processing individual message: {str(e)}"
                                )

                    elif file_path.endswith(".json"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            # Try different possible root keys
                            messages = (
                                data.get("messages")
                                or data.get("sms")
                                or data.get("mms")
                                or data.get("chats")
                                or data.get("instant_messages")
                                or []
                            )

                            for message in messages:
                                try:
                                    content = (
                                        message.get("body")
                                        or message.get("text")
                                        or message.get("content")
                                        or message.get("message")
                                    )

                                    timestamp = (
                                        message.get("timestamp")
                                        or message.get("date")
                                        or message.get("time")
                                    )

                                    sender = (
                                        message.get("from")
                                        or message.get("sender")
                                        or message.get("source")
                                        or message.get("address")
                                    )

                                    receiver = (
                                        message.get("to")
                                        or message.get("recipient")
                                        or message.get("dest")
                                        or message.get("destination")
                                    )

                                    # Create record matching CSV format
                                    record = {
                                        "Preview Text": content,  # Key field for analytics
                                        "From": sender,
                                        "To": receiver,
                                        "Date": timestamp,
                                        "Case Name": os.path.basename(
                                            os.path.dirname(file_path)
                                        ),
                                        "Source": "UFDR",
                                        "Message Type": "message",
                                        "Application": "SMS/MMS",  # Required for analytics
                                        "source_file": os.path.basename(file_path),
                                        "case_id": ObjectId(self.case_id),
                                        "processed": False,
                                        "alert": None,
                                        "analysis_summary": {  # Initialize empty analytics fields
                                            "top_topic": None,
                                            "sentiment_aspect": None,
                                            "emotion": None,
                                            "toxicity_score": None,
                                            "risk_level": None,
                                            "language": None,
                                            "interaction_type": None,
                                            "entities": [],
                                            "entities_classification": {},
                                        },
                                        "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                    }

                                    if record[
                                        "Preview Text"
                                    ]:  # Only insert if there's content
                                        messages_batch.append(record)
                                        inserted_count += 1
                                        if len(messages_batch) >= BATCH_FLUSH_SIZE:
                                            await self.mongo_collection.insert_many(messages_batch, ordered=False)
                                            logger.info(f"Bulk inserted {len(messages_batch)} messages")
                                            messages_batch = []

                                except Exception as e:
                                    logger.error(
                                        f"Error processing individual message: {str(e)}"
                                    )

                except Exception as e:
                    logger.error(f"Error processing message file {file_path}: {str(e)}")

            # Flush remaining messages
            if messages_batch:
                await self.mongo_collection.insert_many(messages_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(messages_batch)} messages")

            # Update total messages count
            new_total = current_count + inserted_count
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)}, {"$set": {"total_messages": new_total}}
            )

            logger.info(f"Successfully processed {inserted_count} messages")

        except Exception as e:
            logger.error(f"Error in process_messages: {str(e)}")
            raise

    async def process_chats(self):
        """Extract and store chats from the UFDR file."""
        try:
            chat_files = self.find_data_files("chat")
            inserted_count = 0
            chats_batch = []
            BATCH_FLUSH_SIZE = 500

            # Update total messages count for chats
            current_total = await self.collection_all_cases.find_one(
                {"_id": ObjectId(self.case_id)}, {"total_messages": 1}
            )
            current_count = (
                current_total.get("total_messages", 0) if current_total else 0
            )

            for file_path in chat_files:
                try:
                    logger.info(f"Processing chat file: {file_path}")

                    if file_path.endswith(".txt"):
                        try:
                            with open(
                                file_path, "r", encoding="utf-8", errors="ignore"
                            ) as f:
                                content = f.read()

                            # Extract chat metadata
                            chat_metadata = {}
                            lines = content.split("\n")

                            # Process metadata section
                            for line in lines:
                                if (
                                    ": " in line
                                    and not line.startswith("From:")
                                    and not line.startswith("To:")
                                ):
                                    key, value = line.split(": ", 1)
                                    chat_metadata[key] = value.strip()

                            # Split content into messages
                            messages = content.split("-----------------------------")

                            for message_block in messages:
                                if not message_block.strip():
                                    continue

                                message_lines = message_block.strip().split("\n")
                                message_data = {}

                                body_started = False
                                body_content = []

                                for line in message_lines:
                                    if line.startswith("Body:"):
                                        body_started = True
                                        continue

                                    if body_started:
                                        body_content.append(line)
                                    elif ": " in line:
                                        key, value = line.split(": ", 1)
                                        message_data[key.strip()] = value.strip()

                                if body_content:
                                    # Parse From/To fields
                                    sender = ""
                                    for field in message_data:
                                        if field.startswith("From"):
                                            sender = message_data[field]
                                            if "@" in sender:
                                                sender = sender.split("@")[
                                                    0
                                                ]  # Remove @domain part
                                            break

                                    # Get chat application type
                                    chat_app = (
                                        message_data.get("Source App")
                                        or message_data.get("Application")
                                        or chat_metadata.get("Application")
                                        or "Chat"
                                    )

                                    # Create record matching CSV format
                                    record = {
                                        "Preview Text": "\n".join(body_content),
                                        "From": sender,
                                        "To": chat_metadata.get("Participants", ""),
                                        "Date": message_data.get("Timestamp", ""),
                                        "Case Name": os.path.basename(
                                            os.path.dirname(os.path.dirname(file_path))
                                        ),
                                        "Source": "UFDR",
                                        "Message Type": "chat",
                                        "Application": chat_app,  # Required for analytics
                                        "Chat App": chat_app,
                                        "Chat Session": chat_metadata.get(
                                            "Start Time", ""
                                        ),
                                        "case_id": ObjectId(self.case_id),
                                        "processed": False,
                                        "alert": None,
                                        "analysis_summary": {  # Initialize empty analytics fields
                                            "top_topic": None,
                                            "sentiment_aspect": None,
                                            "emotion": None,
                                            "toxicity_score": None,
                                            "risk_level": None,
                                            "language": None,
                                            "interaction_type": None,
                                            "entities": [],
                                            "entities_classification": {},
                                        },
                                        "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                    }

                                    if record[
                                        "Preview Text"
                                    ]:  # Only insert if there's content
                                        chats_batch.append(record)
                                        inserted_count += 1
                                        if len(chats_batch) >= BATCH_FLUSH_SIZE:
                                            await self.mongo_collection.insert_many(chats_batch, ordered=False)
                                            logger.info(f"Bulk inserted {len(chats_batch)} chat messages")
                                            chats_batch = []

                        except Exception as e:
                            logger.error(
                                f"Error processing TXT chat file {file_path}: {str(e)}"
                            )
                            continue

                    elif file_path.endswith(".xml"):
                        tree = ET.parse(file_path)
                        root = tree.getroot()

                        # Try different XML structures for chats
                        chat_elements = (
                            root.findall(".//chat")
                            or root.findall(".//message")
                            or root.findall(".//conversation")
                            or root.findall(".//im")
                        )

                        for chat in chat_elements:
                            try:
                                # Extract chat content
                                content = (
                                    chat.findtext("body")
                                    or chat.findtext("text")
                                    or chat.findtext("content")
                                    or chat.findtext("message")
                                )

                                # Extract participants
                                from_participant = (
                                    chat.findtext("from")
                                    or chat.findtext("sender")
                                    or chat.findtext("author")
                                )

                                to_participant = (
                                    chat.findtext("to")
                                    or chat.findtext("recipient")
                                    or chat.findtext("participants")
                                )

                                # Extract timestamp
                                timestamp = (
                                    chat.findtext("timestamp")
                                    or chat.findtext("date")
                                    or chat.findtext("time")
                                )

                                # Extract application if available
                                chat_app = (
                                    chat.findtext("application")
                                    or chat.findtext("app")
                                    or chat.findtext("platform")
                                    or "Chat"
                                )

                                record = {
                                    "Preview Text": content,
                                    "From": from_participant,
                                    "To": to_participant,
                                    "Date": timestamp,
                                    "Case Name": os.path.basename(
                                        os.path.dirname(file_path)
                                    ),
                                    "Source": "UFDR",
                                    "Message Type": "chat",
                                    "Application": chat_app,  # Required for analytics
                                    "Chat App": chat_app,
                                    "chat_id": chat.get("id"),
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "analysis_summary": {  # Initialize empty analytics fields
                                        "top_topic": None,
                                        "sentiment_aspect": None,
                                        "emotion": None,
                                        "toxicity_score": None,
                                        "risk_level": None,
                                        "language": None,
                                        "interaction_type": None,
                                        "entities": [],
                                        "entities_classification": {},
                                    },
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }

                                if record[
                                    "Preview Text"
                                ]:  # Only insert if there's content
                                    chats_batch.append(record)
                                    inserted_count += 1
                                    if len(chats_batch) >= BATCH_FLUSH_SIZE:
                                        await self.mongo_collection.insert_many(chats_batch, ordered=False)
                                        logger.info(f"Bulk inserted {len(chats_batch)} chat messages")
                                        chats_batch = []

                            except Exception as e:
                                logger.error(
                                    f"Error processing individual chat: {str(e)}"
                                )

                    elif file_path.endswith(".json"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            chats = (
                                data.get("chats")
                                or data.get("messages")
                                or data.get("conversations")
                                or []
                            )

                            for chat in chats:
                                try:
                                    content = (
                                        chat.get("body")
                                        or chat.get("text")
                                        or chat.get("content")
                                        or chat.get("message")
                                    )

                                    from_participant = (
                                        chat.get("from")
                                        or chat.get("sender")
                                        or chat.get("author")
                                    )

                                    to_participant = (
                                        chat.get("to")
                                        or chat.get("recipient")
                                        or chat.get("participants")
                                    )

                                    timestamp = (
                                        chat.get("timestamp")
                                        or chat.get("date")
                                        or chat.get("time")
                                    )

                                    chat_app = (
                                        chat.get("application")
                                        or chat.get("app")
                                        or chat.get("platform")
                                        or "Chat"
                                    )

                                    record = {
                                        "Preview Text": content,
                                        "From": from_participant,
                                        "To": to_participant,
                                        "Date": timestamp,
                                        "Case Name": os.path.basename(
                                            os.path.dirname(file_path)
                                        ),
                                        "Source": "UFDR",
                                        "Message Type": "chat",
                                        "Application": chat_app,  # Required for analytics
                                        "Chat App": chat_app,
                                        "chat_id": chat.get("id"),
                                        "case_id": ObjectId(self.case_id),
                                        "processed": False,
                                        "alert": None,
                                        "analysis_summary": {  # Initialize empty analytics fields
                                            "top_topic": None,
                                            "sentiment_aspect": None,
                                            "emotion": None,
                                            "toxicity_score": None,
                                            "risk_level": None,
                                            "language": None,
                                            "interaction_type": None,
                                            "entities": [],
                                            "entities_classification": {},
                                        },
                                        "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                    }

                                    if record[
                                        "Preview Text"
                                    ]:  # Only insert if there's content
                                        chats_batch.append(record)
                                        inserted_count += 1
                                        if len(chats_batch) >= BATCH_FLUSH_SIZE:
                                            await self.mongo_collection.insert_many(chats_batch, ordered=False)
                                            logger.info(f"Bulk inserted {len(chats_batch)} chat messages")
                                            chats_batch = []

                                except Exception as e:
                                    logger.error(
                                        f"Error processing individual chat: {str(e)}"
                                    )

                except Exception as e:
                    logger.error(f"Error processing chat file {file_path}: {str(e)}")
                    continue

            # Flush remaining chat records
            if chats_batch:
                await self.mongo_collection.insert_many(chats_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(chats_batch)} chat messages")

            # Update total messages count
            new_total = current_count + inserted_count
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)}, {"$set": {"total_messages": new_total}}
            )

            logger.info(f"Successfully processed {inserted_count} chat messages")

        except Exception as e:
            logger.error(f"Error in process_chats: {str(e)}")
            raise

    async def process_contacts(self):
        """Extract and store contacts from the UFDR file."""
        try:
            contact_files = self.find_data_files("contact")
            inserted_count = 0
            contacts_batch = []
            BATCH_FLUSH_SIZE = 500

            for file_path in contact_files:
                try:
                    # Handle different formats
                    if file_path.endswith(".xml"):
                        tree = ET.parse(file_path)
                        root = tree.getroot()
                        for contact in root.findall(".//contact"):
                            try:
                                record = {
                                    "name": contact.findtext("name"),
                                    "phone_number": contact.findtext("phone")
                                    or contact.findtext("phoneNumber"),
                                    "email": contact.findtext("email"),
                                    "type": "contact",
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }

                                contacts_batch.append(record)
                                inserted_count += 1
                                if len(contacts_batch) >= BATCH_FLUSH_SIZE:
                                    await self.mongo_collection.insert_many(contacts_batch, ordered=False)
                                    logger.info(f"Bulk inserted {len(contacts_batch)} contacts")
                                    contacts_batch = []

                            except Exception as e:
                                logger.error(
                                    f"Error processing individual contact: {str(e)}"
                                )

                    elif file_path.endswith(".json"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            contacts = data.get("contacts", [])
                            for contact in contacts:
                                try:
                                    record = {
                                        "name": contact.get("name"),
                                        "phone_number": contact.get("phone")
                                        or contact.get("phoneNumber"),
                                        "email": contact.get("email"),
                                        "type": "contact",
                                        "case_id": ObjectId(self.case_id),
                                        "processed": False,
                                        "alert": None,
                                        "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                    }

                                    contacts_batch.append(record)
                                    inserted_count += 1
                                    if len(contacts_batch) >= BATCH_FLUSH_SIZE:
                                        await self.mongo_collection.insert_many(contacts_batch, ordered=False)
                                        logger.info(f"Bulk inserted {len(contacts_batch)} contacts")
                                        contacts_batch = []

                                except Exception as e:
                                    logger.error(
                                        f"Error processing individual contact: {str(e)}"
                                    )

                except Exception as e:
                    logger.error(f"Error processing contact file {file_path}: {str(e)}")

            # Flush remaining contacts
            if contacts_batch:
                await self.mongo_collection.insert_many(contacts_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(contacts_batch)} contacts")

            logger.info(f"Successfully processed {inserted_count} contacts")

        except Exception as e:
            logger.error(f"Error in process_contacts: {str(e)}")
            raise

    async def process_emails(self):
        """Extract and store emails from the UFDR file."""
        try:
            email_files = self.find_data_files("email")
            inserted_count = 0
            emails_batch = []
            BATCH_FLUSH_SIZE = 500

            # Set total messages count initially to 0
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)}, {"$set": {"total_messages": 0}}
            )

            for file_path in email_files:
                try:
                    logger.info(f"Processing email file: {file_path}")

                    if file_path.endswith(".eml"):
                        try:
                            with open(file_path, "rb") as fp:
                                msg = BytesParser(policy=policy.default).parse(fp)

                                # Extract attachments if any
                                attachments = []
                                body = ""

                                if msg.is_multipart():
                                    for part in msg.walk():
                                        if (
                                            part.get_content_disposition()
                                            == "attachment"
                                        ):
                                            attachments.append(
                                                {
                                                    "filename": part.get_filename(),
                                                    "content_type": part.get_content_type(),
                                                }
                                            )
                                        elif part.get_content_type() == "text/plain":
                                            body = part.get_payload(decode=True).decode(
                                                errors="ignore"
                                            )
                                else:
                                    body = msg.get_payload(decode=True).decode(
                                        errors="ignore"
                                    )

                                # Parse addresses
                                from_addr = msg.get("from", "")
                                to_addrs = msg.get_all("to", [])
                                if isinstance(to_addrs, list):
                                    to_addrs = "; ".join(to_addrs)

                                # Format date
                                date = msg.get("date", "")

                                # Create record matching CSV format
                                record = {
                                    "Preview Text": body,  # Main content field matching CSV format
                                    "From": from_addr,
                                    "To": to_addrs,
                                    "Date": date,
                                    "Case Name": os.path.basename(
                                        os.path.dirname(file_path)
                                    ),  # Using parent folder name as case name
                                    "Source": "UFDR",
                                    "Message Type": "email",
                                    "Attachments": attachments,
                                    "Subject": msg.get("subject", ""),
                                    "case_id": ObjectId(self.case_id),
                                    "processed": False,
                                    "alert": None,
                                    "ingestion_timestamp": datetime.datetime.now().isoformat(),
                                }

                                if record[
                                    "Preview Text"
                                ]:  # Only insert if there's content
                                    emails_batch.append(record)
                                    inserted_count += 1
                                    if len(emails_batch) >= BATCH_FLUSH_SIZE:
                                        await self.mongo_collection.insert_many(emails_batch, ordered=False)
                                        logger.info(f"Bulk inserted {len(emails_batch)} emails")
                                        emails_batch = []
                                        await self.collection_all_cases.find_one_and_update(
                                            {"_id": ObjectId(self.case_id)},
                                            {
                                                "$set": {
                                                    "total_messages": inserted_count
                                                }
                                            },
                                        )

                        except Exception as e:
                            logger.error(
                                f"Error processing EML file {file_path}: {str(e)}"
                            )
                            continue

                    elif file_path.endswith((".xml", ".json")):
                        # Similar processing for XML and JSON files...
                        logger.info(f"Skipping non-EML file: {file_path}")
                        continue

                except Exception as e:
                    logger.error(f"Error processing email file {file_path}: {str(e)}")
                    continue

            # Flush remaining emails
            if emails_batch:
                await self.mongo_collection.insert_many(emails_batch, ordered=False)
                logger.info(f"Bulk inserted final {len(emails_batch)} emails")

            # Update final count
            await self.collection_all_cases.find_one_and_update(
                {"_id": ObjectId(self.case_id)},
                {"$set": {"total_messages": inserted_count}},
            )

            logger.info(f"Successfully processed {inserted_count} emails")

        except Exception as e:
            logger.error(f"Error in process_emails: {str(e)}")
            raise

    async def ingest_rag_data(self):
        """Process the ingested data for RAG analysis."""
        try:
            logger.info("Starting RAG data ingestion...")

            # Get embedded vector size from model profile
            embedding_obj = self.models_profile.get("embeddings", {})
            vector_size = embedding_obj.get("embedding_size", 512)

            # Get all documents that haven't been processed for RAG
            data = await self.mongo_collection.find(
                {"processed": False, "Preview Text": {"$exists": True, "$ne": None}}
            ).to_list(None)

            logger.info(f"Found {len(data)} documents for RAG processing")

            if not data:
                logger.info("No documents to process for RAG")
                return

            # Initialize RAG analyzer
            from rag_v1 import ArabicRagAnalyzer

            analyzer = ArabicRagAnalyzer(
                self.collection_all_cases,
                self.mongo_collection,
                self.case_id,
                self.models_profile,
            )

            # Create Qdrant collection
            new_qdrant_collection = f"case_{self.case_id}"
            logger.info(f"Setting up Qdrant collection: {new_qdrant_collection}")

            from qdrant_client.models import Distance, VectorParams

            if not qdrant_client.collection_exists(
                collection_name=new_qdrant_collection
            ):
                qdrant_client.create_collection(
                    collection_name=new_qdrant_collection,
                    vectors_config=VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                )

            # Process using BATCHED embeddings for GPU-optimal throughput
            from config.settings import settings
            compute = settings.compute_config
            embedding_batch_size = compute["embedding_batch_size"]
            qdrant_upsert_batch_size = min(embedding_batch_size, 500)
            logger.info(
                f"RAG BATCH processing: embedding_batch={embedding_batch_size}, "
                f"qdrant_upsert_batch={qdrant_upsert_batch_size}, "
                f"GPU={compute['gpu_name']} ({compute['gpu_memory_gb']}GB VRAM)"
            )

            # Step 1: Collect all texts and metadata
            valid_messages = []
            all_texts = []
            for message in data:
                text = message.get("Preview Text")
                if text and isinstance(text, str) and text.strip():
                    valid_messages.append(message)
                    all_texts.append(text.strip())
                else:
                    logger.warning(f"Skipping message {message.get('_id')}: no valid text")

            if not valid_messages:
                logger.info("No valid messages with text for RAG embedding")
                return

            logger.info(f"Generating batch embeddings for {len(all_texts)} texts...")

            # Step 2: Generate all embeddings in batches using the embedding client directly
            embedding_client = analyzer.embedding_client
            try:
                all_embeddings = embedding_client.create_embeddings_batch(
                    all_texts, batch_size=embedding_batch_size
                )
                logger.info(f"Batch embedding complete: {len(all_embeddings)} embeddings generated")
            except Exception as emb_error:
                logger.error(f"Batch embedding failed: {emb_error}, falling back to individual")
                all_embeddings = []
                for text in all_texts:
                    try:
                        emb = embedding_client.create_embeddings([text])
                        all_embeddings.extend(emb)
                    except Exception as e:
                        logger.error(f"Individual embedding failed: {e}")
                        all_embeddings.append([0.0] * vector_size)

            # Step 3: Create Qdrant points and batch-upsert
            from qdrant_client.models import PointStruct
            current_batch = []
            total_upserted = 0

            for i, (message, embedding) in enumerate(zip(valid_messages, all_embeddings)):
                try:
                    normalized = analyzer.normalize_embedding(np.array(embedding))
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        payload={
                            "message": message.get("Preview Text"),
                            "mongo_id": str(message.get("_id")),
                            "from": message.get("From"),
                            "to": message.get("To"),
                            "date": message.get("Date"),
                            "case_name": message.get("Case Name"),
                        },
                        vector=normalized.tolist() if hasattr(normalized, 'tolist') else list(normalized),
                    )
                    current_batch.append(point)

                    # Upload batch if full
                    if len(current_batch) >= qdrant_upsert_batch_size:
                        try:
                            qdrant_client.upsert(
                                collection_name=new_qdrant_collection,
                                points=current_batch,
                            )
                            total_upserted += len(current_batch)
                            current_batch = []
                        except Exception as batch_error:
                            logger.error(f"Error uploading batch to Qdrant: {str(batch_error)}")
                            for point in current_batch:
                                try:
                                    qdrant_client.upsert(
                                        collection_name=new_qdrant_collection,
                                        points=[point],
                                    )
                                    total_upserted += 1
                                except Exception as point_error:
                                    logger.error(f"Error uploading point to Qdrant: {str(point_error)}")
                            current_batch = []
                except Exception as e:
                    logger.error(f"Error creating point for message {message.get('_id')}: {str(e)}")

            # Upload any remaining points
            if current_batch:
                try:
                    qdrant_client.upsert(
                        collection_name=new_qdrant_collection, points=current_batch
                    )
                    total_upserted += len(current_batch)
                except Exception as e:
                    logger.error(f"Error uploading final batch to Qdrant: {str(e)}")
                    for point in current_batch:
                        try:
                            qdrant_client.upsert(
                                collection_name=new_qdrant_collection, points=[point]
                            )
                            total_upserted += 1
                        except Exception as point_error:
                            logger.error(f"Error uploading point to Qdrant: {str(point_error)}")

            logger.info(f"RAG batch upsert complete: {total_upserted}/{len(valid_messages)} points")

            logger.info("RAG data ingestion completed successfully")

        except Exception as e:
            logger.error(f"Error during RAG ingestion: {str(e)}")
            raise
