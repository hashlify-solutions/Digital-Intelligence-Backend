#!/usr/bin/env python3
"""
Script to create database indexes for UFDR shared collections.
Run this script once to set up optimal indexes for the shared collections approach.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.db import (
    ufdr_calls_collection,
    ufdr_chats_collection,
    ufdr_emails_collection,
    ufdr_locations_collection,
    ufdr_notes_collection,
    ufdr_searched_items_collection,
    ufdr_user_accounts_collection,
    ufdr_audio_collection,
    ufdr_photos_collection,
    ufdr_videos_collection,
    ufdr_files_collection,
    ufdr_photo_detected_faces_collection,
    ufdr_video_detected_faces_collection,
    ufdr_photo_detected_objects_collection,
    ufdr_video_detected_objects_collection,
    ufdr_video_screenshots_collection,
    detectors_collection,
    detector_matches_collection,
    detector_settings_collection,
)


async def create_ufdr_indexes():
    """Create indexes for all UFDR collections for optimal performance"""
    
    print("Creating indexes for UFDR shared collections...")

    # UFDR files collection indexes
    print(f"Creating indexes for {ufdr_files_collection.name}...")
    await ufdr_files_collection.create_index([("caseId", 1), ("created_at", -1)])
    await ufdr_files_collection.create_index([("caseId", 1)])
    await ufdr_files_collection.create_index([("name", 1)])
    await ufdr_files_collection.create_index([("file_size", 1)])
    await ufdr_files_collection.create_index([("associated_schema_names", 1)])
    print(f"✅ Indexes created for {ufdr_files_collection.name}")
    
    # Common indexes for all JSON collections
    json_collections = [
        ufdr_calls_collection,
        ufdr_chats_collection,
        ufdr_emails_collection,
        ufdr_locations_collection,
        ufdr_notes_collection,
        ufdr_searched_items_collection,
        ufdr_user_accounts_collection
    ]
    
    for collection in json_collections:
        print(f"Creating indexes for {collection.name}...")
        
        # Primary query indexes
        await collection.create_index([("ufdr_id", 1), ("created_at", -1)])
        await collection.create_index([("case_id", 1), ("created_at", -1)])
        await collection.create_index([("ufdr_id", 1), ("case_id", 1)])
        
        # Individual field indexes
        await collection.create_index([("ufdr_id", 1)])
        await collection.create_index([("case_id", 1)])
        await collection.create_index([("created_at", -1)])
        
        print(f"✅ Indexes created for {collection.name}")
    
    # Media collections indexes
    media_collections = [ufdr_audio_collection, ufdr_photos_collection, ufdr_videos_collection]
    
    for collection in media_collections:
        print(f"Creating indexes for {collection.name}...")
        
        # Primary query indexes
        await collection.create_index([("ufdr_id", 1), ("created_at", -1)])
        await collection.create_index([("case_id", 1), ("media_type", 1)])
        await collection.create_index([("ufdr_id", 1), ("media_type", 1)])
        
        # Individual field indexes
        await collection.create_index([("ufdr_id", 1)])
        await collection.create_index([("case_id", 1)])
        await collection.create_index([("media_type", 1)])
        await collection.create_index([("type", 1)])
        await collection.create_index([("name", 1)])
        
        print(f"✅ Indexes created for {collection.name}")

    # Video screenshots collection indexes
    print(f"Creating indexes for {ufdr_video_screenshots_collection.name}...")
    
    # Primary query indexes
    await ufdr_video_screenshots_collection.create_index([("ufdr_id", 1), ("created_at", -1)])
    await ufdr_video_screenshots_collection.create_index([("case_id", 1), ("media_type", 1)])
    await ufdr_video_screenshots_collection.create_index([("ufdr_video_id", 1), ("frame_number", 1)])
    await ufdr_video_screenshots_collection.create_index([("ufdr_id", 1), ("ufdr_video_id", 1)])
    
    # Individual field indexes
    await ufdr_video_screenshots_collection.create_index([("ufdr_id", 1)])
    await ufdr_video_screenshots_collection.create_index([("case_id", 1)])
    await ufdr_video_screenshots_collection.create_index([("ufdr_video_id", 1)])
    await ufdr_video_screenshots_collection.create_index([("frame_number", 1)])
    await ufdr_video_screenshots_collection.create_index([("media_type", 1)])
    await ufdr_video_screenshots_collection.create_index([("type", 1)])
    await ufdr_video_screenshots_collection.create_index([("name", 1)])
    
    print(f"✅ Indexes created for {ufdr_video_screenshots_collection.name}")

    # Detected faces collections indexes
    detected_faces_collections = [ufdr_photo_detected_faces_collection, ufdr_video_detected_faces_collection]
    for collection in detected_faces_collections:
        print(f"Creating indexes for {collection.name}...")
        await collection.create_index([("ufdr_id", 1)])
        await collection.create_index([("case_id", 1)])
        if collection.name == ufdr_photo_detected_faces_collection.name:
            await collection.create_index([("ufdr_photo_id", 1)])
        elif collection.name == ufdr_video_detected_faces_collection.name:
            await collection.create_index([("ufdr_video_id", 1)])
        await collection.create_index([("created_at", -1)])
        print(f"✅ Indexes created for {collection.name}")

    # Detected objects collections indexes
    detected_objects_collections = [ufdr_photo_detected_objects_collection, ufdr_video_detected_objects_collection]
    for collection in detected_objects_collections:
        print(f"Creating indexes for {collection.name}...")
        await collection.create_index([("ufdr_id", 1)])
        await collection.create_index([("case_id", 1)])
        if collection.name == ufdr_photo_detected_objects_collection.name:
            await collection.create_index([("ufdr_photo_id", 1)])
        elif collection.name == ufdr_video_detected_objects_collection.name:
            await collection.create_index([("ufdr_video_id", 1)])
        await collection.create_index([("created_at", -1)])
        print(f"✅ Indexes created for {collection.name}")

    # Detectors collection indexes
    print(f"Creating indexes for {detectors_collection.name}...")
    await detectors_collection.create_index([("case_id", 1), ("type", 1)])
    await detectors_collection.create_index([("case_id", 1), ("created_at", -1)])
    await detectors_collection.create_index([("user_id", 1), ("case_id", 1)])
    await detectors_collection.create_index([("case_id", 1)])
    await detectors_collection.create_index([("type", 1)])
    await detectors_collection.create_index([("name", 1)])
    await detectors_collection.create_index([("created_at", -1)])
    print(f"✅ Indexes created for {detectors_collection.name}")

    # Detector matches collection indexes
    print(f"Creating indexes for {detector_matches_collection.name}...")
    await detector_matches_collection.create_index([("case_id", 1), ("similarity_score", -1)])
    await detector_matches_collection.create_index([("detector_id", 1), ("similarity_score", -1)])
    await detector_matches_collection.create_index([("detected_item_id", 1), ("detected_item_type", 1)])
    await detector_matches_collection.create_index([("case_id", 1), ("confidence_level", 1)])
    await detector_matches_collection.create_index([("case_id", 1), ("created_at", -1)])
    await detector_matches_collection.create_index([("case_id", 1)])
    await detector_matches_collection.create_index([("detector_id", 1)])
    await detector_matches_collection.create_index([("detected_item_type", 1)])
    await detector_matches_collection.create_index([("confidence_level", 1)])
    print(f"✅ Indexes created for {detector_matches_collection.name}")

    # Detector settings collection indexes
    print(f"Creating indexes for {detector_settings_collection.name}...")
    await detector_settings_collection.create_index([("case_id", 1)], unique=True)
    await detector_settings_collection.create_index([("user_id", 1)])
    print(f"✅ Indexes created for {detector_settings_collection.name}")
    
    print("\n🎉 All UFDR indexes created successfully!")
    print("\n📊 Performance optimizations:")
    print("  ✅ Fast queries by ufdr_id")
    print("  ✅ Fast queries by case_id") 
    print("  ✅ Fast time-based queries")
    print("  ✅ Fast media type filtering")
    print("  ✅ Fast cross-UFDR analytics")


async def show_collection_stats():
    """Show statistics for all UFDR collections"""
    
    print("\n📈 Collection Statistics:")
    print("=" * 50)
    
    collections = [
        ("UFDR Files", ufdr_files_collection),
        ("Calls", ufdr_calls_collection),
        ("Chats", ufdr_chats_collection),
        ("Emails", ufdr_emails_collection),
        ("Locations", ufdr_locations_collection),
        ("Notes", ufdr_notes_collection),
        ("Searched Items", ufdr_searched_items_collection),
        ("User Accounts", ufdr_user_accounts_collection),
        ("Audio Files", ufdr_audio_collection),
        ("Photos", ufdr_photos_collection),
        ("Videos", ufdr_videos_collection),
        ("Video Screenshots", ufdr_video_screenshots_collection),
        ("Photo Detected Faces", ufdr_photo_detected_faces_collection),
        ("Video Detected Faces", ufdr_video_detected_faces_collection),
        ("Photo Detected Objects", ufdr_photo_detected_objects_collection),
        ("Video Detected Objects", ufdr_video_detected_objects_collection),
        ("Detectors", detectors_collection),
        ("Detector Matches", detector_matches_collection),
        ("Detector Settings", detector_settings_collection),
    ]
    
    for name, collection in collections:
        try:
            count = await collection.count_documents({})
            print(f"{name:15}: {count:,} documents")
        except Exception as e:
            print(f"{name:15}: Error - {e}")


if __name__ == "__main__":
    async def main():
        await create_ufdr_indexes()
        await show_collection_stats()
    
    asyncio.run(main())
