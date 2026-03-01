import asyncio
from config.db import detector_matches_collection

async def delete_all_detector_matches():
    """
    Deletes all documents from the detector matches collections.
    Returns a summary of the deletion operation.
    """
    deletion_summary = {
        "detector_matches_deleted": 0,
        "errors": []
    }

    try:
        # Delete from detector matches collection
        result = await detector_matches_collection.delete_many({})
        deletion_summary["detector_matches_deleted"] = result.deleted_count

    except Exception as e:
        deletion_summary["errors"].append(str(e))

    return deletion_summary

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(delete_all_detector_matches())
    print(f"Deletion Summary:")
    print(f"Detector matches deleted: {result['detector_matches_deleted']}")
    if result["errors"]:
        print("Errors encountered:")
        for error in result["errors"]:
            print(f"- {error}")