from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import os
import logging
from dotenv import load_dotenv
from bson import ObjectId
from config.db import db, collection_case
from config.db import get_neo
from utils.auth import get_current_user
from config.settings import settings
from utils.neo4j_client import Neo4jClient

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


router = APIRouter()


INSERT_CASE = """
MERGE (c:Case {id: $case.id})
  ON CREATE SET c.name = $case.name, c.createdAt = datetime()
  ON MATCH  SET c.name = coalesce($case.name, c.name)
"""

INSERT_PERSON_AND_LINKS = """
MERGE (p:Person {id: $person.id})
  ON CREATE SET p.name = $person.name, p.normalized = toLower($person.name), p.toxicity_score = $person.toxicity_score, p.risk_level = $person.risk_level
  ON MATCH  SET p.name = coalesce($person.name, p.name), p.toxicity_score = coalesce($person.toxicity_score, p.toxicity_score), p.risk_level = coalesce($person.risk_level, p.risk_level)
WITH p
MATCH (c:Case {id: $case_id})
MERGE (p)-[:BELONGS_TO_CASE]->(c)
WITH p
UNWIND $emails AS email
  MERGE (e:Email {address: email})
  MERGE (p)-[:HAS_EMAIL]->(e)
WITH p
UNWIND $phones AS phone
  MERGE (n:PhoneNumber {number: phone})
  MERGE (p)-[:HAS_PHONE]->(n)
"""

INSERT_DEVICE_AND_LINK = """
MERGE (d:Device {id: $device.id})
  ON CREATE SET d.type = $device.type, d.platform = $device.platform
  ON MATCH  SET d.type = coalesce($device.type, d.type),
             d.platform = coalesce($device.platform, d.platform)
WITH d
MATCH (c:Case {id: $case_id})
MERGE (d)-[:BELONGS_TO_CASE]->(c)
"""

LINK_PERSON_DEVICE = """
MATCH (p:Person {id: $person_id})
MATCH (d:Device {id: $device_id})
MERGE (p)-[:OWNS]->(d)
"""

INSERT_CHAT_MESSAGE = """
MERGE (m:ChatMessage {id: $msg.id})
  ON CREATE SET m.app = $msg.app, m.timestamp = datetime($msg.timestamp), m.preview = $msg.preview, m.toxicity_score = $msg.toxicity_score, m.risk_level = $msg.risk_level
  ON MATCH  SET m.app = coalesce($msg.app, m.app),
             m.timestamp = coalesce(datetime($msg.timestamp), m.timestamp),
             m.preview = coalesce($msg.preview, m.preview),
             m.toxicity_score = coalesce($msg.toxicity_score, m.toxicity_score),
             m.risk_level = coalesce($msg.risk_level, m.risk_level)
WITH m
MATCH (fromP:Person {id: $msg.from})
MERGE (fromP)-[:SENT]->(m)
WITH m
UNWIND $to_list AS toId
  MATCH (toP:Person {id: toId})
  MERGE (m)-[:RECEIVED_BY]->(toP)
WITH m
MATCH (c:Case {id: $case_id})
MERGE (m)-[:BELONGS_TO_CASE]->(c)
"""

INSERT_CALL = """
MERGE (c:Call {id: $call.id})
  ON CREATE SET c.timestamp = datetime($call.timestamp), c.direction = $call.direction, c.duration = $call.duration
  ON MATCH  SET c.timestamp = coalesce(datetime($call.timestamp), c.timestamp),
             c.direction = coalesce($call.direction, c.direction),
             c.duration  = coalesce($call.duration, c.duration)
WITH c
MATCH (fromP:Person {id: $call.from})
MERGE (fromP)-[:PLACED]->(c)
WITH c
MATCH (toP:Person {id: $call.to})
MERGE (c)-[:RECEIVED_BY]->(toP)
"""

INSERT_MESSAGE = """
MERGE (m:Message {id: $msg.id})
  ON CREATE SET 
    m.content = $msg.content,
    m.timestamp = datetime($msg.timestamp),
    m.app = $msg.app,
    m.message_type = $msg.message_type,
    m.preview = $msg.preview,
    m.toxicity_score = $msg.toxicity_score,
    m.risk_level = $msg.risk_level
  ON MATCH SET
    m.content = coalesce($msg.content, m.content),
    m.timestamp = coalesce(datetime($msg.timestamp), m.timestamp),
    m.app = coalesce($msg.app, m.app),
    m.message_type = coalesce($msg.message_type, m.message_type),
    m.preview = coalesce($msg.preview, m.preview),
    m.toxicity_score = coalesce($msg.toxicity_score, m.toxicity_score),
    m.risk_level = coalesce($msg.risk_level, m.risk_level)
WITH m
MATCH (fromP:Person {id: $msg.from})
MERGE (fromP)-[:SENT]->(m)
WITH m
UNWIND $to_list AS toId
  MATCH (toP:Person {id: toId})
  MERGE (m)-[:RECEIVED_BY]->(toP)
WITH m
MATCH (c:Case {id: $case_id})
MERGE (m)-[:BELONGS_TO_CASE]->(c)
"""

INSERT_EMAIL = """
MERGE (e:Email {id: $email.id})
  ON CREATE SET 
    e.subject = $email.subject,
    e.timestamp = datetime($email.timestamp),
    e.preview = $email.preview,
    e.from_address = $email.from_address,
    e.to_addresses = $email.to_addresses
  ON MATCH SET
    e.subject = coalesce($email.subject, e.subject),
    e.timestamp = coalesce(datetime($email.timestamp), e.timestamp),
    e.preview = coalesce($email.preview, e.preview),
    e.from_address = coalesce($email.from_address, e.from_address),
    e.to_addresses = coalesce($email.to_addresses, e.to_addresses)
WITH e
MATCH (fromP:Person {id: $email.from_person_id})
MERGE (fromP)-[:SENT_EMAIL]->(e)
WITH e
UNWIND $to_person_ids AS toPersonId
  MATCH (toP:Person {id: toPersonId})
  MERGE (e)-[:SENT_TO]->(toP)
WITH e
MATCH (c:Case {id: $case_id})
MERGE (e)-[:BELONGS_TO_CASE]->(c)
"""

INSERT_LOCATION = """
MERGE (l:Location {id: $location.id})
  ON CREATE SET 
    l.latitude = $location.latitude,
    l.longitude = $location.longitude,
    l.elevation = $location.elevation,
    l.timestamp = datetime($location.timestamp),
    l.source = $location.source,
    l.address = $location.address
  ON MATCH SET
    l.latitude = coalesce($location.latitude, l.latitude),
    l.longitude = coalesce($location.longitude, l.longitude),
    l.elevation = coalesce($location.elevation, l.elevation),
    l.timestamp = coalesce(datetime($location.timestamp), l.timestamp),
    l.source = coalesce($location.source, l.source),
    l.address = coalesce($location.address, l.address)
WITH l
MATCH (p:Person {id: $location.person_id})
MERGE (p)-[:LOCATED_AT]->(l)
WITH l
MATCH (c:Case {id: $case_id})
MERGE (l)-[:BELONGS_TO_CASE]->(c)
"""

INC_COMM_EDGE = """
// Ensure sender person exists and belongs to case
MERGE (a:Person {id: $from_id})
MERGE (c:Case {id: $case_id})
MERGE (a)-[:BELONGS_TO_CASE]->(c)

// Ensure recipient person exists and belongs to case
MERGE (b:Person {id: $to_id})
MERGE (b)-[:BELONGS_TO_CASE]->(c)

// Create communication relationship
MERGE (a)-[r:COMMUNICATED_WITH {channel: $channel, case_id: $case_id}]->(b)
ON CREATE SET r.count = 1
ON MATCH  SET r.count = r.count + 1
"""

Q_PERSONS_IN_MULTI_CASES = """
MATCH (p:Person)-[:BELONGS_TO_CASE]->(c:Case)
WITH p, collect(DISTINCT c.id) AS caseIds
WHERE size(caseIds) > 1
RETURN p.id AS person_id, p.name AS name, size(caseIds) AS case_count, caseIds
ORDER BY case_count DESC, name ASC
"""


def _normalize_person_from_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    pid = str(doc.get("From") or doc.get("Name") or doc.get("email") or doc.get("phone_numbers") or doc.get("_id"))
    name = str(doc.get("Name") or doc.get("From") or "Unknown")
    emails = []
    if doc.get("email"):
        emails = [doc["email"]] if isinstance(doc["email"], str) else list(doc["email"])
    phones = []
    if doc.get("phone_numbers"):
        phones = [doc["phone_numbers"]] if isinstance(doc["phone_numbers"], str) else list(doc["phone_numbers"])
    
    # Extract toxicity and risk data from analysis_summary
    analysis_summary = doc.get("analysis_summary", {})
    toxicity_score = analysis_summary.get("toxicity_score")
    risk_level = analysis_summary.get("risk_level")
    
    return {"id": f"P:{pid}", "name": name, "emails": emails, "phones": phones, "toxicity_score": toxicity_score, "risk_level": risk_level}


def _normalize_person_id(value: Any) -> str:
    return f"P:{str(value)}"


def _normalize_chat(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize chat data for Neo4j insertion"""
    mid = str(doc.get("_id"))
    app = str(doc.get("Application") or doc.get("app") or "Chat")
    ts = str(doc.get("timestamp") or doc.get("Date") or "1970-01-01T00:00:00Z")
    preview = str(doc.get("Preview Text") or "")
    
    # Handle missing From field - use a default identifier
    from_value = doc.get("From") or doc.get("Name") or doc.get("sender")
    if not from_value:
        # Try to extract from other fields
        from_value = doc.get("email") or doc.get("phone_numbers") or "unknown_sender"
    from_id = _normalize_person_id(from_value)
    
    to_field = doc.get("To")
    to_list: List[str] = []
    if isinstance(to_field, str):
        to_list = [f"P:{t.strip()}" for t in to_field.split(",") if t.strip()]
    elif isinstance(to_field, list):
        to_list = [f"P:{str(t)}" for t in to_field if t]  # Filter out empty values
    
    # Extract toxicity and risk data from analysis_summary
    analysis_summary = doc.get("analysis_summary", {})
    toxicity_score = analysis_summary.get("toxicity_score")
    risk_level = analysis_summary.get("risk_level")
    
    return {"id": f"M:{mid}", "app": app, "timestamp": ts, "from": from_id, "to": to_list, "preview": preview, "toxicity_score": toxicity_score, "risk_level": risk_level}


def _normalize_call(doc: Dict[str, Any]) -> Dict[str, Any]:
    cid = str(doc.get("_id"))
    ts = str(doc.get("timestamp") or doc.get("Date") or "1970-01-01T00:00:00Z")
    direction = str(doc.get("direction") or doc.get("Direction") or "unknown")
    duration = int(doc.get("duration") or doc.get("Duration") or 0)
    from_id = _normalize_person_id(doc.get("From") or doc.get("caller"))
    to_id = _normalize_person_id(doc.get("To") or doc.get("callee") or "unknown")
    return {"id": f"C:{cid}", "timestamp": ts, "from": from_id, "to": to_id, "direction": direction, "duration": duration}


def _normalize_message(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize message data for Neo4j insertion"""
    msg_id = str(doc.get("_id"))
    content = str(doc.get("Preview Text") or doc.get("Content") or doc.get("Text") or "")
    timestamp = str(doc.get("Date") or doc.get("Timestamp") or "1970-01-01T00:00:00Z")
    app = str(doc.get("Application") or doc.get("App") or "Unknown")
    msg_type = str(doc.get("Message Type") or "message").lower()
    preview = str(doc.get("Preview Text") or content[:100] if content else "")
    
    # Handle missing From field - use a default identifier
    from_value = doc.get("From") or doc.get("Sender") or doc.get("Name")
    if not from_value:
        # Try to extract from other fields
        from_value = doc.get("email") or doc.get("phone_numbers") or "unknown_sender"
    from_id = _normalize_person_id(from_value)
    
    to_field = doc.get("To") or doc.get("Recipients")
    to_list = []
    if isinstance(to_field, str):
        to_list = [f"P:{t.strip()}" for t in to_field.split(",") if t.strip()]
    elif isinstance(to_field, list):
        to_list = [f"P:{str(t)}" for t in to_field if t]  # Filter out empty values
    
    # Extract toxicity and risk data from analysis_summary
    analysis_summary = doc.get("analysis_summary", {})
    toxicity_score = analysis_summary.get("toxicity_score")
    risk_level = analysis_summary.get("risk_level")
    
    return {
        "id": f"MSG:{msg_id}",
        "content": content,
        "timestamp": timestamp,
        "app": app,
        "message_type": msg_type,
        "preview": preview,
        "from": from_id,
        "to": to_list,
        "toxicity_score": toxicity_score,
        "risk_level": risk_level
    }


def _normalize_email(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize email data for Neo4j insertion"""
    email_id = str(doc.get("_id"))
    subject = str(doc.get("Subject") or "No Subject")
    timestamp = str(doc.get("Date") or doc.get("Timestamp") or "1970-01-01T00:00:00Z")
    preview = str(doc.get("Preview Text") or doc.get("Body") or "")[:200]
    
    # Handle missing From field - use a default identifier
    from_address = str(doc.get("From") or doc.get("From Address") or "")
    from_value = from_address or doc.get("Sender") or "unknown_sender"
    from_person_id = _normalize_person_id(from_value)
    
    to_field = doc.get("To") or doc.get("To Addresses")
    to_addresses = []
    to_person_ids = []
    
    if isinstance(to_field, str):
        to_addresses = [t.strip() for t in to_field.split(",") if t.strip()]
        to_person_ids = [f"P:{t.strip()}" for t in to_field.split(",") if t.strip()]
    elif isinstance(to_field, list):
        to_addresses = [str(t) for t in to_field if t]  # Filter out empty values
        to_person_ids = [f"P:{str(t)}" for t in to_field if t]  # Filter out empty values
    
    return {
        "id": f"EMAIL:{email_id}",
        "subject": subject,
        "timestamp": timestamp,
        "preview": preview,
        "from_address": from_address,
        "to_addresses": to_addresses,
        "from_person_id": from_person_id,
        "to_person_ids": to_person_ids
    }


def _normalize_location(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize location data for Neo4j insertion"""
    loc_id = str(doc.get("_id"))
    latitude = float(doc.get("Latitude") or doc.get("lat") or 0.0)
    longitude = float(doc.get("Longitude") or doc.get("lng") or 0.0)
    elevation = float(doc.get("Elevation") or doc.get("alt") or 0.0)
    timestamp = str(doc.get("Date") or doc.get("Timestamp") or "1970-01-01T00:00:00Z")
    source = str(doc.get("Source") or doc.get("Application") or "Unknown")
    address = str(doc.get("Address") or doc.get("Location") or "")
    
    person_id = _normalize_person_id(doc.get("Person") or doc.get("Device") or doc.get("From") or "unknown")
    
    return {
        "id": f"LOC:{loc_id}",
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
        "timestamp": timestamp,
        "source": source,
        "address": address,
        "person_id": person_id
    }


def _sync_case_docs(neo_client: Neo4jClient, case: Dict[str, Any], docs: List[Dict[str, Any]]) -> None:
    """Synchronize case documents to Neo4j with proper validation and ordering."""
    try:
        # Insert case node first
        neo_client.run(INSERT_CASE, {"case": case})

        # Collect all unique people from both senders and recipients
        seen_person_ids = set()
        
        # First pass: collect all people from senders (From, Name, email, phone_numbers)
        for d in docs:
            for key in ["From", "Name", "email", "phone_numbers"]:
                if d.get(key):
                    p = _normalize_person_from_doc(d)
                    seen_person_ids.add(p["id"])
        
        # Second pass: collect all people from recipients (To field)
        for d in docs:
            to_field = d.get("To")
            if to_field:
                if isinstance(to_field, str):
                    to_list = [t.strip() for t in to_field.split(",") if t.strip()]
                elif isinstance(to_field, list):
                    to_list = [str(t) for t in to_field]
                else:
                    to_list = []
                
                for to_person in to_list:
                    person_id = _normalize_person_id(to_person)
                    seen_person_ids.add(person_id)
        
        # Create all Person nodes first - this ensures they exist before message processing
        created_person_ids = set()
        
        # Create Person nodes from documents (with full details)
        for d in docs:
            for key in ["From", "Name", "email", "phone_numbers"]:
                if d.get(key):
                    p = _normalize_person_from_doc(d)
                    if p["id"] not in created_person_ids:
                        try:
                            neo_client.run(INSERT_PERSON_AND_LINKS, {
                                "person": p,
                                "case_id": case["id"],
                                "emails": p.get("emails", []),
                                "phones": p.get("phones", [])
                            })
                            created_person_ids.add(p["id"])
                        except Exception as e:
                            print(f"Warning: Failed to create person node {p['id']}: {e}")
        
        # Create Person nodes for recipients (with minimal details)
        for person_id in seen_person_ids:
            if person_id not in created_person_ids:
                try:
                    # Extract just the identifier part (remove "P:" prefix)
                    identifier = person_id[2:] if person_id.startswith("P:") else person_id
                    recipient_person = {
                        "id": person_id,
                        "name": identifier,  # Use the identifier as the name
                        "emails": [],
                        "phones": []
                    }
                    neo_client.run(INSERT_PERSON_AND_LINKS, {
                        "person": recipient_person,
                        "case_id": case["id"],
                        "emails": [],
                        "phones": []
                    })
                    created_person_ids.add(person_id)
                except Exception as e:
                    print(f"Warning: Failed to create recipient person node {person_id}: {e}")

        # Validate person creation before processing messages
        if not created_person_ids:
            print(f"Warning: No person nodes created for case {case['id']}")

        # Process Messages - now with guaranteed person node existence
        message_count = 0
        for d in docs:
            msg_type = str(d.get("Message Type") or "").lower()
            try:
                if msg_type in ["message", "chat", "sms", "sms/mms"]:
                    m = _normalize_chat(d)
                    neo_client.run(INSERT_CHAT_MESSAGE, {"msg": m, "to_list": m.get("to", []), "case_id": case["id"]})
                    for to_id in m.get("to", []):
                        if to_id in created_person_ids and m["from"] in created_person_ids:
                            neo_client.run(INC_COMM_EDGE, {"from_id": m["from"], "to_id": to_id, "channel": m.get("app", "chat"), "case_id": case["id"]})
                        else:
                            print(f"Warning: Skipping communication edge - person nodes not found: {m['from']} -> {to_id}")
                    message_count += 1
                elif msg_type == "call":
                    c = _normalize_call(d)
                    neo_client.run(INSERT_CALL, {"call": c})
                    if c["from"] in created_person_ids and c["to"] in created_person_ids:
                        neo_client.run(INC_COMM_EDGE, {"from_id": c["from"], "to_id": c["to"], "channel": "call", "case_id": case["id"]})
                    else:
                        print(f"Warning: Skipping call edge - person nodes not found: {c['from']} -> {c['to']}")
                    message_count += 1
                else:
                    # Handle any other message types that might exist
                    # Check if this is a message-like document by looking for common message fields
                    if d.get("Application") or d.get("app") or d.get("From") or d.get("sender"):
                        # Treat it as a message and create relationships
                        m = _normalize_chat(d)
                        neo_client.run(INSERT_CHAT_MESSAGE, {"msg": m, "to_list": m.get("to", []), "case_id": case["id"]})
                        for to_id in m.get("to", []):
                            if to_id in created_person_ids and m["from"] in created_person_ids:
                                neo_client.run(INC_COMM_EDGE, {"from_id": m["from"], "to_id": to_id, "channel": m.get("app", "unknown"), "case_id": case["id"]})
                            else:
                                print(f"Warning: Skipping communication edge - person nodes not found: {m['from']} -> {to_id}")
                        message_count += 1
            except Exception as e:
                print(f"Warning: Failed to process message document: {e}")
        
        # Process Messages as separate nodes
        for d in docs:
            if str(d.get("Message Type") or "").lower() in ["message", "chat", "sms", "sms/mms"] or d.get("Application"):
                try:
                    msg = _normalize_message(d)
                    neo_client.run(INSERT_MESSAGE, {"msg": msg, "to_list": msg["to"], "case_id": case["id"]})
                except Exception as e:
                    print(f"Warning: Failed to create message node: {e}")
        
        # Process Emails as separate nodes
        email_count = 0
        for d in docs:
            if str(d.get("Message Type") or "").lower() == "email" or d.get("Subject"):
                try:
                    email = _normalize_email(d)
                    neo_client.run(INSERT_EMAIL, {"email": email, "to_person_ids": email["to_person_ids"], "case_id": case["id"]})
                    email_count += 1
                except Exception as e:
                    print(f"Warning: Failed to create email node: {e}")
        
        # Process Locations as separate nodes
        location_count = 0
        for d in docs:
            if d.get("Latitude") or d.get("Longitude") or d.get("Location"):
                try:
                    loc = _normalize_location(d)
                    neo_client.run(INSERT_LOCATION, {"location": loc, "case_id": case["id"]})
                    location_count += 1
                except Exception as e:
                    print(f"Warning: Failed to create location node: {e}")
        
        print(f"Case {case['id']} sync completed: {len(created_person_ids)} persons, {message_count} messages, {email_count} emails, {location_count} locations")
        
    except Exception as e:
        print(f"Error synchronizing case {case['id']}: {e}")
        raise


@router.post("/sync/{case_id}")
async def sync_case(case_id: str, _: str = Depends(get_current_user)):
    try:
        case_doc = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case_doc:
            raise HTTPException(status_code=404, detail="Case not found")
        case_name = case_doc.get("name")
        collection = db[f"{case_name}_{case_id}"]
        docs = await collection.find({}).to_list(None)

        case_obj = {"id": str(case_doc["_id"]), "name": case_name}
        neo_client = get_neo()
        _sync_case_docs(neo_client, case_obj, docs)
        return {"status": "ok", "nodes": len(docs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-case-persons")
async def cross_case_persons(_: str = Depends(get_current_user)):
    try:
        neo_client = get_neo()
        res = neo_client.run(Q_PERSONS_IN_MULTI_CASES)
        return {"data": [r.data() for r in res]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cleanup-all")
async def cleanup_all_neo4j_data(_: str = Depends(get_current_user)):
    """
    Delete all synced data from Neo4j database.
    This will remove all nodes and relationships created by the sync process.
    WARNING: This operation cannot be undone.
    """
    try:
        neo_client = get_neo()
        
        # First, get a count of what will be deleted
        count_result = neo_client.run("""
            MATCH (n)
            WHERE n:Case OR n:Person OR n:ChatMessage OR n:Call OR n:Message OR 
                   n:Email OR n:Location OR n:Device OR n:PhoneNumber
            RETURN count(n) as total_nodes
        """)
        total_nodes = list(count_result)[0]["total_nodes"] if count_result else 0
        
        # Delete all synced data
        # Use DETACH DELETE to automatically remove relationships and avoid constraint violations
        delete_result = neo_client.run("""
            MATCH (n)
            WHERE n:Case OR n:Person OR n:ChatMessage OR n:Call OR 
                  n:Message OR n:Email OR n:Location OR n:Device OR n:PhoneNumber
            DETACH DELETE n
            RETURN 'Cleanup completed successfully' as message
        """)
        
        message = list(delete_result)[0]["message"] if delete_result else "Cleanup completed"
        
        logger.info(f"Neo4j cleanup completed. Deleted {total_nodes} nodes and all relationships.")
        
        return {
            "message": message,
            "deleted_nodes": total_nodes,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup Neo4j data: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.delete("/cleanup-case/{case_id}")
async def cleanup_case_data(case_id: str, _: str = Depends(get_current_user)):
    """
    Delete all data for a specific case from Neo4j.
    This will remove the case node, all related person nodes, messages, and relationships.
    """
    try:
        neo_client = get_neo()
        
        # First check if the case exists and belongs to the user
        case_check = await collection_case.find_one({
            "_id": ObjectId(case_id),
            "user_id": ObjectId(_)
        })
        
        if not case_check:
            raise HTTPException(status_code=404, detail="Case not found or not owned by user")
        
        # Get count of nodes that will be deleted for this case
        count_result = neo_client.run("""
            MATCH (c:Case {id: $case_id})
            OPTIONAL MATCH (c)<-[:BELONGS_TO_CASE]-(n)
            RETURN count(n) + 1 as total_nodes
        """, {"case_id": case_id})
        
        total_nodes = list(count_result)[0]["total_nodes"] if count_result else 0
        
        # Delete all data for this specific case
        delete_result = neo_client.run("""
            // Find the case and all related nodes, then detach delete everything
            MATCH (c:Case {id: $case_id})
            OPTIONAL MATCH (c)<-[:BELONGS_TO_CASE]-(related)
            DETACH DELETE related, c
            RETURN 'Case cleanup completed successfully' as message
        """, {"case_id": case_id})
        
        message = list(delete_result)[0]["message"] if delete_result else "Case cleanup completed"
        
        logger.info(f"Case {case_id} cleanup completed. Deleted {total_nodes} nodes and all relationships.")
        
        return {
            "message": message,
            "case_id": case_id,
            "deleted_nodes": total_nodes,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cleanup case {case_id} data: {e}")
        raise HTTPException(status_code=500, detail=f"Case cleanup failed: {str(e)}")


# Canned, safe queries with optional case scoping
_QUERIES: Dict[str, str] = {
    # Persons that appear in more than one case (optionally within a subset of cases)
    "cross_case_persons": """
    WITH $caseIds AS caseIds
    MATCH (p:Person)-[:BELONGS_TO_CASE]->(c:Case)
    WHERE caseIds IS NULL OR c.id IN caseIds
    WITH p, collect(DISTINCT c.id) AS caseIdsList
    WHERE size(caseIdsList) > 1
    RETURN p.id AS person_id, p.name AS name, size(caseIdsList) AS case_count, caseIdsList
    ORDER BY case_count DESC, name ASC
    """,
    # Ego network for a person limited to selected cases
    "ego_network_in_cases": """
    WITH $caseIds AS caseIds
    MATCH (p:Person {id: $person_id})
    OPTIONAL MATCH (p)-[:BELONGS_TO_CASE]->(cp:Case)
    WITH p, collect(DISTINCT cp.id) AS allCases, caseIds
    OPTIONAL MATCH (p)-[r:COMMUNICATED_WITH]->(o:Person)
    WITH p, r, o, caseIds
    OPTIONAL MATCH (p)-[:BELONGS_TO_CASE]->(cp2:Case)
    WHERE caseIds IS NULL OR cp2.id IN caseIds
    OPTIONAL MATCH (o)-[:BELONGS_TO_CASE]->(co:Case)
    WHERE caseIds IS NULL OR co.id IN caseIds
    RETURN p{.*, cases: collect(DISTINCT cp2.id)} AS person,
           collect(DISTINCT o{.*, cases: coalesce(collect(DISTINCT co.id), [])}) AS contacts,
           collect(DISTINCT r{.*}) AS edges
    """,
}


@router.post("/query")
async def run_canned_query(payload: Dict[str, Any], _: str = Depends(get_current_user)):
    try:
        key = str(payload.get("query_key"))
        if key not in _QUERIES:
            raise HTTPException(status_code=400, detail=f"Unsupported query_key: {key}")
        params = payload.get("params", {}) or {}
        case_ids = payload.get("case_ids")
        if case_ids is not None and not isinstance(case_ids, list):
            raise HTTPException(status_code=400, detail="case_ids must be a list of strings or omitted")
        final_params = {**params, "caseIds": case_ids}
        neo_client = get_neo()
        res = neo_client.run(_QUERIES[key], final_params)
        data = [r.data() for r in res]
        return {"data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/case-sync-status")
async def get_case_sync_status(_: str = Depends(get_current_user)):
    """
    Get sync status of all user cases - which ones are in Neo4j vs only in MongoDB.
    This helps users understand which cases are available for graph visualization.
    """
    try:
        # Get all user cases from MongoDB
        mongo_cases = await collection_case.find({"user_id": ObjectId(_)}).to_list(None)
        
        # Get all cases from Neo4j
        neo_client = get_neo()
        neo4j_result = neo_client.run("""
            MATCH (c:Case)
            RETURN c.id as case_id, c.name as case_name
        """)
        neo4j_cases = {record["case_id"]: record["case_name"] for record in neo4j_result}
        
        # Build status for each case
        case_status = []
        for case_doc in mongo_cases:
            case_id = str(case_doc["_id"])
            case_name = case_doc.get("name", "Unnamed Case")
            
            # Check if case exists in Neo4j
            in_neo4j = case_id in neo4j_cases
            
            # Get document count from MongoDB
            collection = db[f"{case_name}_{case_id}"]
            doc_count = await collection.count_documents({})
            
            case_status.append({
                "id": case_id,
                "name": case_name,
                "in_mongodb": True,
                "in_neo4j": in_neo4j,
                "mongo_doc_count": doc_count,
                "sync_status": "synced" if in_neo4j else "not_synced"
            })
        
        return {
            "data": case_status,
            "summary": {
                "total_cases": len(case_status),
                "synced_to_neo4j": sum(1 for c in case_status if c["in_neo4j"]),
                "not_synced": sum(1 for c in case_status if not c["in_neo4j"])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-case-ids")
async def get_user_case_ids(_: str = Depends(get_current_user)):
    try:
        items = await collection_case.find({"user_id": ObjectId(_)}).to_list(None)
        data = [{"id": str(c["_id"]), "name": c.get("name", "Unnamed Case")} for c in items]
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Default graph cypher that returns a {nodes, edges} object
# This query ignores channel/time filters and includes any
# COMMUNICATED_WITH relationships between people who belong to
# the provided cases. It also includes Case nodes and the
# BELONGS_TO_CASE edges linking people to their cases.
DEFAULT_GRAPH_CYPHER = """
WITH $caseIds AS caseIds

// Handle empty caseIds - return empty result
WITH caseIds, CASE WHEN caseIds IS NULL OR size(caseIds) = 0 THEN [] ELSE caseIds END as valid_case_ids

// Get people and cases in the specified cases (with optional matching)
OPTIONAL MATCH (p:Person)-[:BELONGS_TO_CASE]->(c:Case)
WHERE c.id IN valid_case_ids
WITH valid_case_ids, 
     collect(DISTINCT p) as people_in_cases, 
     collect(DISTINCT c) as cases_in

// Get communication relationships between people in these cases (optional)
OPTIONAL MATCH (p1:Person)-[r:COMMUNICATED_WITH]->(p2:Person)
WHERE p1 IN people_in_cases AND p2 IN people_in_cases
WITH valid_case_ids, people_in_cases, cases_in,
     collect(DISTINCT {
         from: p1.id,
         to: p2.id,
         channel: r.channel,
         count: r.count,
         case_id: r.case_id
     }) as comm_relationships

// Get messages in specified cases (optional)
OPTIONAL MATCH (msg:Message)-[:BELONGS_TO_CASE]->(c:Case)
WHERE c.id IN valid_case_ids
WITH valid_case_ids, people_in_cases, cases_in, comm_relationships,
     collect(DISTINCT msg) as messages_in_cases

// Get emails in specified cases (optional)
OPTIONAL MATCH (email:Email)-[:BELONGS_TO_CASE]->(c:Case)
WHERE c.id IN valid_case_ids
WITH valid_case_ids, people_in_cases, cases_in, comm_relationships, messages_in_cases,
     collect(DISTINCT email) as emails_in_cases

// Get locations in specified cases (optional)
OPTIONAL MATCH (location:Location)-[:BELONGS_TO_CASE]->(c:Case)
WHERE c.id IN valid_case_ids
WITH valid_case_ids, people_in_cases, cases_in, comm_relationships, messages_in_cases, emails_in_cases,
     collect(DISTINCT location) as locations_in_cases

// Get chat messages in specified cases (optional)
OPTIONAL MATCH (chat:ChatMessage)-[:BELONGS_TO_CASE]->(c:Case)
WHERE c.id IN valid_case_ids
WITH valid_case_ids, people_in_cases, cases_in, comm_relationships, messages_in_cases, emails_in_cases, locations_in_cases,
     collect(DISTINCT chat) as chat_messages_in_cases

// Get person->case relationships (optional)
OPTIONAL MATCH (person:Person)-[b:BELONGS_TO_CASE]->(caseNode:Case)
WHERE person IN people_in_cases AND caseNode IN cases_in
WITH valid_case_ids, people_in_cases, cases_in, comm_relationships, messages_in_cases, emails_in_cases, locations_in_cases, chat_messages_in_cases,
     collect(DISTINCT {
         from: person.id,
         to: caseNode.id,
         channel: 'belongs',
         count: 1,
         case_id: caseNode.id
     }) as person_case_relationships

// Build final graph structure - handle empty collections and null values
WITH 
  // Handle null collections by providing empty lists
  coalesce(people_in_cases, []) as people_in_cases,
  coalesce(cases_in, []) as cases_in,
  coalesce(comm_relationships, []) as comm_relationships,
  coalesce(messages_in_cases, []) as messages_in_cases,
  coalesce(emails_in_cases, []) as emails_in_cases,
  coalesce(locations_in_cases, []) as locations_in_cases,
  coalesce(chat_messages_in_cases, []) as chat_messages_in_cases,
  coalesce(person_case_relationships, []) as person_case_relationships

RETURN {
    nodes: ( [person IN people_in_cases | {
        id: person.id,
        name: coalesce(person.name, 'Unknown'),
        type: 'person',
        toxicity_score: coalesce(person.toxicity_score, null),
        risk_level: coalesce(person.risk_level, null)
    }] ) + ( [case IN cases_in | {
        id: case.id,
        name: coalesce(case.name, 'Unnamed Case'),
        type: 'case'
    }] ) + ( [msg IN messages_in_cases | {
        id: msg.id,
        name: substring(coalesce(msg.preview, msg.content, ''), 0, 50),
        type: 'message',
        app: coalesce(msg.app, 'Unknown'),
        timestamp: coalesce(msg.timestamp, '1970-01-01T00:00:00Z'),
        toxicity_score: coalesce(msg.toxicity_score, null),
        risk_level: coalesce(msg.risk_level, null)
    }] ) + ( [email IN emails_in_cases | {
        id: email.id,
        name: substring(coalesce(email.subject, ''), 0, 50),
        type: 'email',
        subject: coalesce(email.subject, ''),
        timestamp: coalesce(email.timestamp, '1970-01-01T00:00:00Z')
    }] ) + ( [location IN locations_in_cases | {
        id: location.id,
        name: coalesce(location.address, 'Location'),
        type: 'location',
        latitude: coalesce(location.latitude, 0.0),
        longitude: coalesce(location.longitude, 0.0),
        address: coalesce(location.address, ''),
        timestamp: coalesce(location.timestamp, '1970-01-01T00:00:00Z'),
        source: coalesce(location.source, 'Unknown')
    }] ) + ( [chat IN chat_messages_in_cases | {
        id: chat.id,
        name: substring(coalesce(chat.content, ''), 0, 50),
        type: 'chat_message',
        app: coalesce(chat.app, 'Unknown'),
        timestamp: coalesce(chat.timestamp, '1970-01-01T00:00:00Z'),
        toxicity_score: coalesce(chat.toxicity_score, null),
        risk_level: coalesce(chat.risk_level, null)
    }] ),
    
    edges: ( [rel IN comm_relationships | {
        id: rel.from + '->' + rel.to + '|' + rel.channel,
        type: 'COMM',
        source: rel.from,
        target: rel.to,
        channel: rel.channel,
        count: rel.count,
        case_id: rel.case_id
    }] ) + ( [rel IN person_case_relationships | {
        id: rel.from + '->' + rel.to + '|BELONGS',
        type: 'BELONGS_TO_CASE',
        source: rel.from,
        target: rel.to,
        channel: rel.channel,
        count: rel.count,
        case_id: rel.case_id
    }] )
} AS graph
"""


@router.post("/sync-selected-cases")
async def sync_selected_cases(payload: Dict[str, Any], _: str = Depends(get_current_user)):
    """
    Sync only selected cases from MongoDB to Neo4j for graph visualization.
    This allows users to choose which cases to upload before drawing the graph.
    """
    try:
        selected_case_ids = payload.get("case_ids", [])
        if not selected_case_ids:
            raise HTTPException(status_code=400, detail="No case_ids provided")
        
        if not isinstance(selected_case_ids, list):
            raise HTTPException(status_code=400, detail="case_ids must be a list")
        
        synced_cases = []
        failed_cases = []
        neo_client = get_neo()
        
        for case_id in selected_case_ids:
            try:
                # Get case from MongoDB - ensure it belongs to the current user
                case_doc = await collection_case.find_one({
                    "_id": ObjectId(case_id),
                    "user_id": ObjectId(_)
                })
                if not case_doc:
                    failed_cases.append({"case_id": case_id, "error": "Case not found or not owned by user"})
                    continue
                
                case_name = case_doc.get("name")
                collection = db[f"{case_name}_{case_id}"]
                docs = await collection.find({}).to_list(None)
                
                if not docs:
                    failed_cases.append({"case_id": case_id, "error": "No documents found in case collection"})
                    continue
                
                # Create case object for Neo4j
                case_obj = {"id": str(case_doc["_id"]), "name": case_name}
                
                # Sync to Neo4j
                _sync_case_docs(neo_client, case_obj, docs)
                synced_cases.append({
                    "case_id": case_id,
                    "name": case_name,
                    "documents_synced": len(docs)
                })
                
            except Exception as e:
                failed_cases.append({"case_id": case_id, "error": str(e)})
        
        return {
            "status": "completed",
            "synced_cases": synced_cases,
            "failed_cases": failed_cases,
            "summary": {
                "total_requested": len(selected_case_ids),
                "successfully_synced": len(synced_cases),
                "failed": len(failed_cases)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/graph-query")
async def graph_query(payload: Dict[str, Any], _: str = Depends(get_current_user)):
    try:
        case_ids = payload.get("case_ids")
        if case_ids is not None and not isinstance(case_ids, list):
            raise HTTPException(status_code=400, detail="case_ids must be a list of strings or omitted")
        # channels = payload.get("channels") or ["chat", "call"]
        # if channels is not None and not isinstance(channels, list):
        #     raise HTTPException(status_code=400, detail="channels must be a list of strings")
        params = payload.get("params", {}) or {}
        cypher_override = payload.get("cypher")

        neo_client = get_neo()
        
        # Test connection first
        try:
            test_result = neo_client.run("RETURN 1 as test")
            list[Any](test_result)  # Consume the result
        except Exception as conn_error:
            raise HTTPException(
                status_code=503, 
                detail=f"Neo4j connection failed: {str(conn_error)}. Please ensure Neo4j is running and accessible at {settings.neo4j_uri}"
            )

        # If case_ids provided, check which ones are already in Neo4j and sync missing ones
        sync_summary = None
        if case_ids:
            # Check which cases are already in Neo4j
            existing_cases_result = neo_client.run("""
                MATCH (c:Case)
                WHERE c.id IN $case_ids
                RETURN c.id as case_id
            """, {"case_ids": case_ids})
            existing_case_ids = [record["case_id"] for record in existing_cases_result]
            
            # Find cases that need to be synced
            missing_case_ids = [case_id for case_id in case_ids if case_id not in existing_case_ids]
            
            # Sync missing cases
            if missing_case_ids:
                sync_result = await sync_selected_cases({"case_ids": missing_case_ids}, _)
                sync_summary = sync_result

        final_params = {
            "caseIds": case_ids,
        }
        
        query_text = cypher_override or DEFAULT_GRAPH_CYPHER
        res = neo_client.run(query_text, final_params)
        rows = [r.data() for r in res]

        # If using default cypher, rows = [{graph: {nodes, edges}}]
        response_data = {"data": rows[0]["graph"]} if rows and "graph" in rows[0] else {"data": rows}
        
        # Add sync summary if cases were synced
        if sync_summary:
            response_data["sync_summary"] = sync_summary
            
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

