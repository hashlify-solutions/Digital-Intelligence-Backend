# Case API Specification

This document provides a comprehensive specification for all case-related API endpoints.

## Base Information
- **Base URL**: `http://localhost:8000/case`
- **Authentication**: Bearer Token (JWT) required for all endpoints
- **Content-Type**: `application/json` or `multipart/form-data` (for file uploads)

---

## Endpoints

### 1. Upload Case Data (CSV)

**Endpoint**: `POST /upload-data`

**Description**: Uploads a CSV file and creates a new case for data processing

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data
```

**Request Type**: `multipart/form-data`

**Form Data Fields**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file (UploadFile) | ✅ Mandatory | CSV file to upload |
| topics | string | ✅ Mandatory | Comma-separated list of topics |
| sentiments | string | ✅ Mandatory | Comma-separated list of sentiments |
| interactions | string | ✅ Mandatory | Comma-separated list of interactions |
| entitiesClasses | string | ✅ Mandatory | Comma-separated list of entity classes |
| caseName | string | ✅ Mandatory | Name for the case |
| category | string | ✅ Mandatory | Case category |
| is_rag | boolean | ✅ Mandatory | Whether to use RAG processing |
| alert_id | string | ❌ Optional | Associated alert ID |
| models_profile_id | string | ❌ Optional | Processing profile ID (uses default if not provided) |

**Response**:

**Success (200)**:
```json
{
  "message": "File uploaded successfully. Processing started.",
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "task_id": "celery-task-uuid"
}
```

**Error (400)**:
```json
{
  "detail": "Only CSV files are allowed in CSV mode." | "Topics, Sentiments, and Interactions are required for RAG data."
}
```

**Error (404)**:
```json
{
  "detail": "Model Profile not found." | "User not found."
}
```

**Error (500)**:
```json
{
  "detail": "Error uploading case: [specific error message]"
}
```

---

### 2. Upload UFDR Data

**Endpoint**: `POST /upload-ufdr-data`

**Description**: Uploads a UFDR file to an existing case or creates a new case

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data
```

**Request Type**: `multipart/form-data`

**Form Data Fields**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | file (UploadFile) | ✅ Mandatory | UFDR file to upload (.ufdr extension) |
| topics | string | ✅ Mandatory | Comma-separated list of topics |
| sentiments | string | ✅ Mandatory | Comma-separated list of sentiments |
| interactions | string | ✅ Mandatory | Comma-separated list of interactions |
| entitiesClasses | string | ✅ Mandatory | Comma-separated list of entity classes |
| caseName | string | ✅ Mandatory | Name for the case (if creating new) |
| category | string | ✅ Mandatory | Case category |
| is_rag | boolean | ✅ Mandatory | Whether to use RAG processing |
| alert_id | string | ❌ Optional | Associated alert ID |
| models_profile_id | string | ❌ Optional | Processing profile ID |
| case_id | string | ❌ Optional | Existing case ID (creates new case if not provided) |

**Response**:

**Success (200)**:
```json
{
  "message": "UFDR file uploaded successfully. Processing started.",
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "ufdr_file_id": "64f1a2b3c4d5e6f7g8h9i0j2",
  "task_id": "celery-task-uuid",
  "file_type": "ufdr",
  "file_size_bytes": 1073741824,
  "file_size_gb": 1.0,
  "file_path": "/path/to/uploaded/file.ufdr"
}
```

**Error (400)**:
```json
{
  "detail": "Only UFDR files are allowed." | "Invalid case ID format"
}
```

**Error (404)**:
```json
{
  "detail": "Case not found or access denied" | "Model Profile not found."
}
```

---

### 3. Delete Case

**Endpoint**: `DELETE /{case_id}`

**Description**: Deletes a case and all associated data

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case to delete |

**Response**:

**Success (200)**:
```json
{
  "message": "Case deleted successfully"
}
```

**Error (404)**:
```json
{
  "detail": "Case not found or you don't have permission to access it"
}
```

---

### 4. Check Task Status

**Endpoint**: `GET /task-status/{task_id}`

**Description**: Checks the status of a Celery background task

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| task_id | string | ✅ Mandatory | Celery task ID |

**Response**:

**Success (200)**:
```json
{
  "task_id": "celery-task-uuid",
  "status": "SUCCESS" | "PENDING" | "FAILURE",
  "result": "Task result or error message"
}
```

---

### 5. Get All Cases

**Endpoint**: `GET /cases-all`

**Description**: Retrieves all cases for the authenticated user with alert counts

**Authentication**: ✅ Required (Bearer Token)

**Response**:

**Success (200)**:
```json
[
  {
    "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "name": "Case Name",
    "status": "completed",
    "user_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "topics": ["security", "threat"],
    "sentiments": ["negative", "positive"],
    "interactions": ["chat", "email"],
    "is_rag": true,
    "model_profile": "64f1a2b3c4d5e6f7g8h9i0j1",
    "entitiesClasses": ["person", "organization"],
    "category": "security",
    "alert_count": 5
  }
]
```

---

### 6. Get Case by ID

**Endpoint**: `GET /case/{case_id}`

**Description**: Retrieves all messages for a specific case

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Response**:

**Success (200)**:
```json
[
  {
    "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "content": "Message content",
    "analysis_summary": {
      "toxicity_score": 0.75,
      "sentiment_aspect": "negative",
      "emotion": "anger",
      "language": "en",
      "risk_level": "high",
      "top_topic": "security",
      "interaction_type": "chat",
      "entities": ["John", "Company"],
      "entities_classification": {
        "person": ["John"],
        "organization": ["Company"]
      }
    },
    "alert": true
  }
]
```

**Error (404)**:
```json
{
  "detail": "Case not found" | "Collection not found"
}
```

---

### 7. Get Case (Paginated)

**Endpoint**: `GET /case-paginated/{case_id}`

**Description**: Retrieves messages for a case with pagination

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| page | integer | ❌ Optional | 1 | Page number (minimum: 1) |
| limit | integer | ❌ Optional | 10 | Items per page (maximum: 100) |

**Response**:

**Success (200)**:
```json
{
  "data": [
    {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
      "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
      "content": "Message content",
      "analysis_summary": {},
      "alert": false
    }
  ],
  "pagination": {
    "total": 1000,
    "page": 1,
    "limit": 10,
    "total_pages": 100
  }
}
```

---

### 8. Get Alert Messages

**Endpoint**: `GET /alert-messages/{case_id}`

**Description**: Retrieves only messages marked as alerts for a case

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Response**: Same format as Get Case by ID, but filtered for alert messages only

---

### 9. Filter Messages

**Endpoint**: `GET /message-filter/{case_id}`

**Description**: Retrieves filtered messages based on various criteria with pagination

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| page | integer | ❌ Optional | 1 | Page number |
| limit | integer | ❌ Optional | 3 | Items per page (max: 100) |
| sematic_search | string | ❌ Optional | null | Semantic search query |
| top_topic | List[string] | ❌ Optional | null | Filter by topics |
| toxicity_score | integer | ❌ Optional | null | Minimum toxicity score |
| sentiment_aspect | List[string] | ❌ Optional | null | Filter by sentiment |
| emotion | List[string] | ❌ Optional | null | Filter by emotions |
| language | List[string] | ❌ Optional | null | Filter by languages |
| risk_level | List[string] | ❌ Optional | null | Filter by risk levels |
| application_type | List[string] | ❌ Optional | null | Filter by applications |
| interaction_type | List[string] | ❌ Optional | null | Filter by interaction types |
| entities | List[string] | ❌ Optional | null | Filter by entities |
| entities_classes | List[string] | ❌ Optional | null | Filter by entity classes |
| alert | boolean | ❌ Optional | null | Filter by alert status |

**Response**: Same paginated format as Get Case (Paginated)

---

### 10. Analyze User RAG Query

**Endpoint**: `POST /analyze-user-rag-query`

**Description**: Analyzes a user query using RAG (Retrieval-Augmented Generation)

**Authentication**: ✅ Required (Bearer Token)

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | Case ID for analysis |
| query | string | ✅ Mandatory | User query to analyze |

**Response**:

**Success (200)**:
```json
{
  "summary": "Generated summary based on the query",
  "mongo_ids": ["64f1a2b3c4d5e6f7g8h9i0j1", "64f1a2b3c4d5e6f7g8h9i0j2"]
}
```

**Error (404)**:
```json
{
  "detail": "Case not found" | "Failed to generate summary."
}
```

---

### 11. Get Analytics

**Endpoint**: `GET /analytics/{case_id}`

**Description**: Retrieves comprehensive analytics data for a case

**Authentication**: Not explicitly required (but recommended)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Response**:

**Success (200)**:
```json
{
  "bar_and_donut": {
    "data": [],
    "chart_type": "bar_and_donut",
    "description": "Bar and Donut chart showing the distribution of topics, sentiments etc."
  },
  "heatmap": {
    "data": [],
    "chart_type": "heatmap",
    "description": "Heatmap showing the distribution of applications against emotions."
  },
  "stacked_bar": {
    "data": [],
    "chart_type": "stacked_bar",
    "description": "Stacked bar chart showing the distribution of risk levels per language."
  },
  "area_chart": {
    "data": [],
    "chart_type": "area_chart",
    "description": "Area chart showing the top 10 most repeated entities"
  },
  "top_cards": {
    "data": [{
      "totalMessages": 1000,
      "highRiskMessages": 50,
      "uniqueUsers": 25,
      "alertMessages": 30,
      "top_3_applications_message_count": {},
      "top_3_entities_classes": {}
    }],
    "chart_type": "top_cards",
    "description": "Top cards showing metrics"
  },
  "side_cards": {
    "data": [],
    "chart_type": "side_cards",
    "description": "Side cards showing various statistics"
  },
  "entity_classes_heatmaps": []
}
```

---

### 12. Get Filtered Analytics

**Endpoint**: `GET /analytics-filtered/{case_id}`

**Description**: Retrieves analytics data with applied filters

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Query Parameters**: Same as Filter Messages endpoint

**Response**: Same format as Get Analytics

---

### 13. Get Messages by IDs

**Endpoint**: `POST /get-messages-by-ids`

**Description**: Retrieves specific messages by their IDs with pagination

**Authentication**: ✅ Required (Bearer Token)

**Request Body**:
```json
{
  "case_id": "string",
  "message_ids": ["string"],
  "page": "integer",
  "limit": "integer"
}
```

**Field Details**:
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| case_id | string | ✅ Mandatory | - | Case ID |
| message_ids | List[string] | ✅ Mandatory | - | Array of message IDs |
| page | integer | ❌ Optional | 1 | Page number (minimum: 1) |
| limit | integer | ❌ Optional | 10 | Items per page (maximum: 100) |

**Response**: Same paginated format as other message endpoints

---

### 14. Get UFDR Files by Case

**Endpoint**: `GET /ufdr-files/{case_id}`

**Description**: Retrieves all UFDR files associated with a case

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Response**:

**Success (200)**:
```json
{
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "ufdr_files": [
    {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j2",
      "caseId": "64f1a2b3c4d5e6f7g8h9i0j1",
      "name": "data.ufdr",
      "file_size": 1073741824,
      "created_at": "2024-12-01T12:00:00.000Z",
      "updated_at": "2024-12-01T12:00:00.000Z"
    }
  ],
  "total_files": 1
}
```

---

### 15. Get UFDR Data

**Endpoint**: `GET /ufdr-data/{ufdr_file_id}/{data_type}`

**Description**: Retrieves specific type of data from a UFDR file with pagination

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| ufdr_file_id | string | ✅ Mandatory | UFDR file ID |
| data_type | string | ✅ Mandatory | Data type (calls, chats, emails, locations, notes, searched_items, user_accounts, audio, photos, videos) |

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| page | integer | ❌ Optional | 1 | Page number |
| limit | integer | ❌ Optional | 10 | Items per page (max: 100) |

**Response**:

**Success (200)**:
```json
{
  "ufdr_file_id": "64f1a2b3c4d5e6f7g8h9i0j2",
  "data_type": "chats",
  "data": [
    {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j3",
      "ufdr_id": "64f1a2b3c4d5e6f7g8h9i0j2",
      "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
      "content": "Chat message content",
      "created_at": "2024-12-01T12:00:00.000Z",
      "updated_at": "2024-12-01T12:00:00.000Z"
    }
  ],
  "pagination": {
    "total": 100,
    "page": 1,
    "limit": 10,
    "total_pages": 10
  }
}
```

**Error (400)**:
```json
{
  "detail": "Invalid data type. Available types: [calls, chats, emails, locations, notes, searched_items, user_accounts, audio, photos, videos]"
}
```

---

### 16. Get UFDR Analytics by Case

**Endpoint**: `GET /ufdr-analytics/{case_id}`

**Description**: Retrieves analytics across all UFDR files for a case

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Response**:

**Success (200)**:
```json
{
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "total_calls": 150,
  "total_chats": 500,
  "total_emails": 75,
  "total_locations": 25,
  "total_notes": 100,
  "total_searched_items": 50,
  "total_user_accounts": 10,
  "total_audio_files": 20,
  "total_photos": 200,
  "total_videos": 30,
  "total_ufdr_files": 3,
  "ufdr_files_summary": [
    {
      "ufdr_file_id": "64f1a2b3c4d5e6f7g8h9i0j2",
      "name": "data.ufdr",
      "file_size_gb": 1.5,
      "associated_schemas": ["chats", "calls"],
      "created_at": "2024-12-01T12:00:00.000Z"
    }
  ]
}
```

---

### 17. Delete UFDR Data

**Endpoint**: `DELETE /ufdr-data/{ufdr_file_id}`

**Description**: Deletes all UFDR data (database and disk) for a specific UFDR file

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| ufdr_file_id | string | ✅ Mandatory | UFDR file ID to delete |

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | Case ID for validation |

**Response**:

**Success (200)**:
```json
{
  "message": "UFDR data deletion completed successfully",
  "deletion_summary": {
    "ufdr_file_id": "64f1a2b3c4d5e6f7g8h9i0j2",
    "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "ufdr_filename": "data.ufdr",
    "deleted_from_collections": ["ufdr_chats", "ufdr_calls"],
    "deleted_document_counts": {
      "ufdr_chats": 500,
      "ufdr_calls": 150
    },
    "disk_cleanup": {
      "success": true,
      "deleted_paths": ["/path/to/ufdr/directory"]
    },
    "errors": []
  },
  "total_documents_deleted": 650,
  "collections_affected": 2,
  "disk_cleanup_success": true,
  "has_errors": false
}
```

---

### 18. Delete All UFDR Data for Case

**Endpoint**: `DELETE /ufdr-data/case/{case_id}/all`

**Description**: Deletes ALL UFDR data for a specific case

**Authentication**: ✅ Required (Bearer Token)

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | Case ID |

**Response**: Similar to Delete UFDR Data but for all UFDR files in the case

---

### 19. Test Analyzer

**Endpoint**: `POST /test-analyzer`

**Description**: Tests the Arabic social analyzer with sample text

**Authentication**: Not required

**Query Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| text | string | ✅ Mandatory | Text to analyze |

**Response**: Analysis results object

---

### 20. Awake All

**Endpoint**: `GET /awake-all`

**Description**: Wakes up the analyzer service (health check/warm-up)

**Authentication**: Not required

**Response**:
```json
"Success"
```

---

## Data Models

### Case Schema
```typescript
{
  _id: string;                            // MongoDB ObjectId
  name: string;                           // Case name
  status: string;                         // Processing status
  user_id: string;                        // Owner user ID
  topics: string[];                       // Array of topics
  sentiments: string[];                   // Array of sentiments
  interactions: string[];                 // Array of interaction types
  is_rag: boolean;                        // RAG processing flag
  model_profile: string;                  // Model profile ID
  entitiesClasses: string[];              // Entity classes
  category: string;                       // Case category
  alert_id?: string;                      // Optional alert ID
  alert_count?: number;                   // Alert count (in responses)
}
```

### Message Schema
```typescript
{
  _id: string;                            // Message ID
  case_id: string;                        // Case ID
  content: string;                        // Message content
  analysis_summary: {
    toxicity_score?: number;              // Toxicity score (0-1)
    sentiment_aspect?: string;            // Sentiment classification
    emotion?: string;                     // Detected emotion
    language?: string;                    // Detected language
    risk_level?: string;                  // Risk assessment
    top_topic?: string;                   // Primary topic
    interaction_type?: string;            // Interaction type
    entities?: string[];                  // Extracted entities
    entities_classification?: object;     // Classified entities
  };
  alert: boolean;                         // Alert flag
  Application?: string;                   // Source application
}
```

### UFDR File Schema
```typescript
{
  _id: string;                            // UFDR file ID
  caseId: string;                         // Associated case ID
  name: string;                           // Original filename
  file_size: number;                      // File size in bytes
  created_at: string;                     // Creation timestamp
  updated_at: string;                     // Update timestamp
  associated_schema_names?: string[];     // Associated data types
}
```

### GetMessagesByIdsRequest Schema
```typescript
{
  case_id: string;                        // Required - Case ID
  message_ids: string[];                  // Required - Array of message IDs
  page?: number;                          // Optional - Page number (default: 1, min: 1)
  limit?: number;                         // Optional - Items per page (default: 10, max: 100)
}
```

---

## Authentication Details

### Required Authentication
Most endpoints require JWT authentication. Include Bearer token in Authorization header.

### Token Usage
```
Authorization: Bearer <your_jwt_token>
```

---

## File Upload Details

### Supported File Types
- **CSV files**: For case data upload (`.csv` extension)
- **UFDR files**: For forensic data upload (`.ufdr` or `.UFDR` extension)

### File Size Handling
- Large files are processed in chunks (1MB chunks)
- Progress logging for files > 1GB
- Disk space validation before upload

---

## Error Handling

### Common HTTP Status Codes
- **200**: Success
- **207**: Multi-Status (partial success with some errors)
- **400**: Bad Request (invalid file type, validation errors)
- **401**: Unauthorized (invalid/expired token)
- **404**: Not Found (case/file not found)
- **422**: Unprocessable Entity (validation errors)
- **500**: Internal Server Error

### Error Response Format
```json
{
  "detail": "Error message description"
}
```

---

## Background Processing

### Celery Tasks
File uploads trigger background Celery tasks for processing:
- **CSV processing**: `process_csv_upload.delay()`
- **UFDR processing**: `process_ufdr_upload.delay()`

### Task Status Monitoring
Use the `/task-status/{task_id}` endpoint to monitor processing progress.

---

## Pagination

### Standard Pagination Parameters
- **page**: Page number (minimum: 1, default: 1)
- **limit**: Items per page (maximum: 100, default varies by endpoint)

### Pagination Response Format
```json
{
  "data": [],
  "pagination": {
    "total": 1000,
    "page": 1,
    "limit": 10,
    "total_pages": 100
  }
}
```

---

## Advanced Features

### Semantic Search
- Available in message filtering endpoints
- Uses RAG (Retrieval-Augmented Generation) for intelligent search
- Requires model profile configuration

### Analytics Pipelines
- Comprehensive analytics with multiple chart types
- Filterable analytics with same filter options as message filtering
- Entity-specific heatmaps and visualizations

### UFDR Data Management
- Support for multiple UFDR files per case
- Granular data type access (calls, chats, emails, etc.)
- Bulk deletion capabilities with detailed reporting

---

## Notes

1. **User Isolation**: All operations are user-specific and access-controlled
2. **File Management**: Automatic file organization and cleanup
3. **Background Processing**: Asynchronous processing for large files
4. **Analytics**: Rich analytics with filtering and visualization support
5. **UFDR Support**: Comprehensive forensic data file support
6. **Error Handling**: Detailed error reporting with specific status codes
7. **Pagination**: Consistent pagination across all list endpoints
8. **Validation**: Comprehensive input validation and sanitization
