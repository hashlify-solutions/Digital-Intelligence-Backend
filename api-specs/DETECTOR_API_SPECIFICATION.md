# Detector API Specification

This document provides a comprehensive specification for all detector-related API endpoints.

## Base Information
- **Base URL**: `http://localhost:8000/detectors`
- **Authentication**: Bearer Token (JWT) required for all endpoints
- **Content-Type**: `application/json` or `multipart/form-data` (for file uploads)

---

## Endpoints

### 1. Upload Detector

**Endpoint**: `POST /case/{case_id}/detectors`

**Description**: Uploads a detector image (person or object) for a specific case and triggers embedding generation

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Request Type**: `multipart/form-data`

**Form Data Fields**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| name | string | ✅ Mandatory | Detector name | 1-100 characters |
| type | string | ✅ Mandatory | Detector type | Must be "person" or "object" |
| description | string | ❌ Optional | Detector description | Max 500 characters |
| file | file (UploadFile) | ✅ Mandatory | Image file | Must be an image (image/*) |

**Response**:

**Success (200)**:
```json
{
  "id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "name": "Person Detector 1",
  "type": "person",
  "description": "Detector for identifying specific person",
  "image_path": "/path/to/detector/image.jpg",
  "has_embedding": false,
  "created_at": "2024-12-01T12:00:00.000Z",
  "updated_at": "2024-12-01T12:00:00.000Z",
  "user_id": "64f1a2b3c4d5e6f7g8h9i0j1"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid case ID format" | "Type must be 'person' or 'object'" | "File must be an image" | "Invalid person image: [validation message]"
}
```

**Error (404)**:
```json
{
  "detail": "Case not found"
}
```

**Error (500)**:
```json
{
  "detail": "Error uploading detector: [specific error message]"
}
```

**Additional Details**:
- Image validation is performed based on detector type
- Embedding generation task is triggered automatically
- Invalid images are rejected and cleaned up
- File names are sanitized for security

---

### 2. List Detectors

**Endpoint**: `GET /case/{case_id}/detectors`

**Description**: Retrieves all detectors for a specific case with optional filtering

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Query Parameters**:
| Parameter | Type | Required | Description | Validation |
|-----------|------|----------|-------------|------------|
| type | string | ❌ Optional | Filter by detector type | Must be "person" or "object" |

**Response**:

**Success (200)**:
```json
[
  {
    "id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "name": "Person Detector 1",
    "type": "person",
    "description": "Detector for identifying specific person",
    "image_path": "/path/to/detector/image.jpg",
    "has_embedding": true,
    "created_at": "2024-12-01T12:00:00.000Z",
    "updated_at": "2024-12-01T12:00:00.000Z",
    "user_id": "64f1a2b3c4d5e6f7g8h9i0j1"
  }
]
```

**Error (400)**:
```json
{
  "detail": "Invalid case ID format"
}
```

**Error (500)**:
```json
{
  "detail": "Error listing detectors: [specific error message]"
}
```

**Additional Details**:
- Results are sorted by creation date (newest first)
- Filtering by type is optional

---

### 3. Get Detector by ID

**Endpoint**: `GET /case/{case_id}/detectors/{detector_id}`

**Description**: Retrieves a specific detector by its ID

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |
| detector_id | string | ✅ Mandatory | MongoDB ObjectId of the detector |

**Response**:

**Success (200)**:
```json
{
  "id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "name": "Person Detector 1",
  "type": "person",
  "description": "Detector for identifying specific person",
  "image_path": "/path/to/detector/image.jpg",
  "has_embedding": true,
  "created_at": "2024-12-01T12:00:00.000Z",
  "updated_at": "2024-12-01T12:00:00.000Z",
  "user_id": "64f1a2b3c4d5e6f7g8h9i0j1"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid ID format"
}
```

**Error (404)**:
```json
{
  "detail": "Detector not found"
}
```

**Error (500)**:
```json
{
  "detail": "Error getting detector: [specific error message]"
}
```

---

### 4. Update Detector

**Endpoint**: `PUT /case/{case_id}/detectors/{detector_id}`

**Description**: Updates detector name and/or description

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |
| detector_id | string | ✅ Mandatory | MongoDB ObjectId of the detector |

**Request Body**:
```json
{
  "name": "string",
  "description": "string"
}
```

**Field Details**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| name | string | ❌ Optional | Updated detector name | 1-100 characters |
| description | string | ❌ Optional | Updated description | Max 500 characters |

**Response**:

**Success (200)**:
```json
{
  "id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "name": "Updated Detector Name",
  "type": "person",
  "description": "Updated description",
  "image_path": "/path/to/detector/image.jpg",
  "has_embedding": true,
  "created_at": "2024-12-01T12:00:00.000Z",
  "updated_at": "2024-12-01T12:30:00.000Z",
  "user_id": "64f1a2b3c4d5e6f7g8h9i0j1"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid ID format"
}
```

**Error (404)**:
```json
{
  "detail": "Detector not found"
}
```

**Error (500)**:
```json
{
  "detail": "Error updating detector: [specific error message]"
}
```

**Additional Details**:
- Only provided fields are updated
- `updated_at` timestamp is automatically set

---

### 5. Delete Detector

**Endpoint**: `DELETE /case/{case_id}/detectors/{detector_id}`

**Description**: Deletes a detector, its associated matches, and image file

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |
| detector_id | string | ✅ Mandatory | MongoDB ObjectId of the detector |

**Response**:

**Success (200)**:
```json
{
  "message": "Detector deleted successfully"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid ID format"
}
```

**Error (404)**:
```json
{
  "detail": "Detector not found"
}
```

**Error (500)**:
```json
{
  "detail": "Error deleting detector: [specific error message]"
}
```

**Additional Details**:
- Deletes detector record from database
- Removes all associated matches
- Deletes physical image file from disk
- Cleanup continues even if file deletion fails

---

### 6. Analyze Detectors

**Endpoint**: `POST /case/{case_id}/detectors/analyze`

**Description**: Triggers analysis of all detectors against detected items in the case

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Request Body**:
```json
{
  "detector_type": "string",
  "recompute_embeddings": "boolean",
  "similarity_threshold_override": "number"
}
```

**Field Details**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| detector_type | string | ❌ Optional | Filter analysis by detector type | Must be "person" or "object" |
| recompute_embeddings | boolean | ❌ Optional | Whether to recompute embeddings | Default: false |
| similarity_threshold_override | number | ❌ Optional | Override similarity threshold | 0.0 to 1.0 |

**Response**:

**Success (200)**:
```json
{
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "analysis_started": true,
  "task_id": "celery-task-uuid",
  "message": "Detector analysis started successfully",
  "detectors_processed": 5,
  "detected_items_to_analyze": 150
}
```

**Error (400)**:
```json
{
  "detail": "Invalid case ID format" | "No detectors with embeddings found for analysis"
}
```

**Error (500)**:
```json
{
  "detail": "Error starting detector analysis: [specific error message]"
}
```

**Additional Details**:
- Only detectors with embeddings are processed
- Counts detected items across photos and videos
- Returns estimated processing counts
- Triggers background Celery task

---

### 7. Get Detector Matches

**Endpoint**: `GET /case/{case_id}/detector-matches`

**Description**: Retrieves detector matches for a case with filtering and statistics

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Query Parameters**:
| Parameter | Type | Required | Default | Description | Validation |
|-----------|------|----------|---------|-------------|------------|
| detector_id | string | ❌ Optional | null | Filter by detector ID | Valid ObjectId |
| confidence_level | string | ❌ Optional | null | Filter by confidence level | "high", "medium", or "low" |
| limit | integer | ❌ Optional | 100 | Maximum matches to return | 1-1000 |

**Response**:

**Success (200)**:
```json
{
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "total_matches": 45,
  "high_confidence_matches": 12,
  "medium_confidence_matches": 20,
  "low_confidence_matches": 13,
  "detector_stats": {
    "person": 30,
    "object": 15
  },
  "matches": [
    {
      "id": "64f1a2b3c4d5e6f7g8h9i0j2",
      "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
      "detector_id": "64f1a2b3c4d5e6f7g8h9i0j3",
      "detector_name": "Person Detector 1",
      "detector_type": "person",
      "detected_item_type": "face",
      "detected_item_id": "64f1a2b3c4d5e6f7g8h9i0j4",
      "detected_item_collection": "ufdr_photo_detected_faces",
      "detected_item_path": "/path/to/detected/face.jpg",
      "similarity_score": 0.95,
      "confidence_level": "high",
      "match_threshold": 0.9,
      "created_at": "2024-12-01T12:00:00.000Z",
      "source_image_path": "/path/to/source/image.jpg",
      "frame_number": null,
      "source_video_path": null
    }
  ]
}
```

**Error (400)**:
```json
{
  "detail": "Invalid case ID format" | "Invalid detector ID format"
}
```

**Error (500)**:
```json
{
  "detail": "Error getting detector matches: [specific error message]"
}
```

**Additional Details**:
- Results are sorted by similarity score (highest first)
- Includes comprehensive statistics
- Supports filtering by detector and confidence level
- Limited to prevent performance issues

---

### 8. Get Detector Settings

**Endpoint**: `GET /case/{case_id}/detector-settings`

**Description**: Retrieves detector analysis settings for a case

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Response**:

**Success (200)**:
```json
{
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "face_thresholds": {
    "high_confidence": 0.9,
    "medium_confidence": 0.75,
    "low_confidence": 0.6,
    "minimum_match": 0.96
  },
  "object_thresholds": {
    "high_confidence": 0.85,
    "medium_confidence": 0.7,
    "low_confidence": 0.55,
    "minimum_match": 0.96
  },
  "user_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "created_at": "2024-12-01T12:00:00.000Z",
  "updated_at": "2024-12-01T12:00:00.000Z"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid case ID format"
}
```

**Error (500)**:
```json
{
  "detail": "Error getting detector settings: [specific error message]"
}
```

**Additional Details**:
- Returns default settings if none exist
- Thresholds control confidence level classification

---

### 9. Update Detector Settings

**Endpoint**: `PUT /case/{case_id}/detector-settings`

**Description**: Updates detector analysis settings for a case

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Request Body**:
```json
{
  "face_thresholds": {
    "high_confidence": "number",
    "medium_confidence": "number",
    "low_confidence": "number",
    "minimum_match": "number"
  },
  "object_thresholds": {
    "high_confidence": "number",
    "medium_confidence": "number",
    "low_confidence": "number",
    "minimum_match": "number"
  }
}
```

**Field Details**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| face_thresholds | object | ❌ Optional | Face detection thresholds | Threshold values 0.0-1.0 |
| object_thresholds | object | ❌ Optional | Object detection thresholds | Threshold values 0.0-1.0 |

**Threshold Object Structure**:
| Field | Type | Description |
|-------|------|-------------|
| high_confidence | number | Threshold for high confidence matches |
| medium_confidence | number | Threshold for medium confidence matches |
| low_confidence | number | Threshold for low confidence matches |
| minimum_match | number | Minimum threshold for any match |

**Response**:

**Success (200)**:
```json
{
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "face_thresholds": {
    "high_confidence": 0.95,
    "medium_confidence": 0.8,
    "low_confidence": 0.65,
    "minimum_match": 0.96
  },
  "object_thresholds": {
    "high_confidence": 0.9,
    "medium_confidence": 0.75,
    "low_confidence": 0.6,
    "minimum_match": 0.96
  },
  "user_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "created_at": "2024-12-01T12:00:00.000Z",
  "updated_at": "2024-12-01T12:30:00.000Z"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid case ID format"
}
```

**Error (500)**:
```json
{
  "detail": "Error updating detector settings: [specific error message]"
}
```

**Additional Details**:
- Uses upsert operation (creates if doesn't exist)
- Only provided threshold objects are updated
- Automatically sets timestamps

---

### 10. Re-analyze Detector Matches

**Endpoint**: `POST /case/{case_id}/detector-matches/reanalyze`

**Description**: Re-runs detector analysis with updated settings, optionally clearing existing matches

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| case_id | string | ✅ Mandatory | MongoDB ObjectId of the case |

**Request Body**:
```json
{
  "detector_type": "string",
  "recompute_embeddings": "boolean",
  "similarity_threshold_override": "number"
}
```

**Field Details**: Same as Analyze Detectors endpoint

**Response**:

**Success (200)**:
```json
{
  "message": "Detector re-analysis started successfully",
  "case_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "task_id": "celery-task-uuid",
  "cleared_existing_matches": true
}
```

**Error (400)**:
```json
{
  "detail": "Invalid case ID format"
}
```

**Error (500)**:
```json
{
  "detail": "Error starting detector re-analysis: [specific error message]"
}
```

**Additional Details**:
- Optionally clears existing matches before re-analysis
- Uses updated settings from detector-settings endpoint
- Triggers background processing task

---

## Data Models

### DetectorResponse Schema
```typescript
{
  id: string;                             // Detector ID (alias for _id)
  case_id: string;                        // Case ID
  name: string;                           // Detector name
  type: "person" | "object";              // Detector type
  description?: string | null;            // Optional description
  image_path: string;                     // Path to detector image
  has_embedding: boolean;                 // Whether embedding is generated
  created_at: string;                     // ISO datetime
  updated_at: string;                     // ISO datetime
  user_id: string;                        // Owner user ID
}
```

### DetectorUpdate Schema
```typescript
{
  name?: string;                          // Optional - Updated name (1-100 chars)
  description?: string;                   // Optional - Updated description (max 500 chars)
}
```

### DetectorMatch Schema
```typescript
{
  id: string;                             // Match ID
  case_id: string;                        // Case ID
  detector_id: string;                    // Detector ID
  detector_name: string;                  // Detector name
  detector_type: "person" | "object";     // Detector type
  detected_item_type: "face" | "object";  // Detected item type
  detected_item_id: string;               // Detected item ID
  detected_item_collection: string;       // Source collection name
  detected_item_path: string;             // Path to detected item
  similarity_score: number;               // Similarity score (0.0-1.0)
  confidence_level: "high" | "medium" | "low"; // Confidence classification
  match_threshold: number;                // Threshold used (0.0-1.0)
  created_at: string;                     // ISO datetime
  source_image_path?: string;             // Optional - Source image path
  frame_number?: number;                  // Optional - Video frame number
  source_video_path?: string;             // Optional - Source video path
}
```

### DetectorMatchSummary Schema
```typescript
{
  case_id: string;                        // Case ID
  total_matches: number;                  // Total number of matches
  high_confidence_matches: number;        // High confidence count
  medium_confidence_matches: number;      // Medium confidence count
  low_confidence_matches: number;         // Low confidence count
  detector_stats: {                       // Matches by detector type
    [detector_type: string]: number;
  };
  matches: DetectorMatch[];               // Array of matches
}
```

### DetectorSettings Schema
```typescript
{
  case_id: string;                        // Case ID
  face_thresholds: {                      // Face detection thresholds
    high_confidence: number;              // High confidence threshold
    medium_confidence: number;            // Medium confidence threshold
    low_confidence: number;               // Low confidence threshold
    minimum_match: number;                // Minimum match threshold
  };
  object_thresholds: {                    // Object detection thresholds
    high_confidence: number;              // High confidence threshold
    medium_confidence: number;            // Medium confidence threshold
    low_confidence: number;               // Low confidence threshold
    minimum_match: number;                // Minimum match threshold
  };
  user_id: string;                        // User ID
  created_at?: string;                    // Optional - ISO datetime
  updated_at?: string;                    // Optional - ISO datetime
}
```

### AnalyzeDetectorsRequest Schema
```typescript
{
  detector_type?: "person" | "object";    // Optional - Filter by detector type
  recompute_embeddings: boolean;          // Whether to recompute embeddings
  similarity_threshold_override?: number; // Optional - Override threshold (0.0-1.0)
}
```

### DetectorAnalysisResponse Schema
```typescript
{
  case_id: string;                        // Case ID
  analysis_started: boolean;              // Whether analysis started
  task_id: string;                        // Celery task ID
  message: string;                        // Status message
  detectors_processed: number;            // Number of detectors to process
  detected_items_to_analyze: number;      // Number of detected items
}
```

---

## Authentication Details

### Required Authentication
All endpoints require JWT authentication:
- Include Bearer token in Authorization header
- Token must be valid and not expired
- User ID is extracted from JWT payload

### Token Usage
```
Authorization: Bearer <your_jwt_token>
```

---

## File Upload Details

### Supported Image Types
- All image formats with `image/*` MIME type
- Images are validated based on detector type
- Files are saved with sanitized names

### File Storage
- Images stored in case-specific directories
- Path format: `{upload_dir}/{case_name}_{case_id}/detectors/`
- Automatic cleanup on deletion

---

## Background Processing

### Embedding Generation
- Triggered automatically on detector upload
- Uses face or object embedding clients
- Sets `has_embedding` flag when complete

### Detector Analysis
- Compares detector embeddings with detected items
- Processes faces and objects separately
- Generates similarity scores and confidence levels

### Celery Tasks
- `process_detector_embedding_task` - Generate embeddings
- `analyze_detector_matches_task` - Analyze matches

---

## Error Handling

### Common HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid format, validation errors)
- **401**: Unauthorized (invalid/expired token)
- **404**: Not Found (detector/case not found)
- **500**: Internal Server Error

### Error Response Format
```json
{
  "detail": "Error message description"
}
```

### Validation Errors
- Image validation failures include specific messages
- ObjectId format validation
- Detector type validation ("person" or "object")
- Confidence level validation ("high", "medium", "low")

---

## Advanced Features

### Image Validation
- Person detectors: Face detection validation
- Object detectors: Object detection validation
- Failed validation triggers cleanup

### Similarity Matching
- Configurable thresholds per detector type
- Three confidence levels with customizable thresholds
- Minimum match threshold to filter weak matches

### Statistics and Analytics
- Match counts by confidence level
- Detector type statistics
- Comprehensive match summaries

### Batch Operations
- Analyze all detectors at once
- Filter analysis by detector type
- Re-analysis with setting updates

---

## Usage Examples

### Upload Detector (cURL)
```bash
curl -X POST "http://localhost:8000/detectors/case/64f1a2b3c4d5e6f7g8h9i0j1/detectors" \
  -H "Authorization: Bearer your_jwt_token" \
  -F "name=Person Detector 1" \
  -F "type=person" \
  -F "description=Detector for John Doe" \
  -F "file=@detector_image.jpg"
```

### Analyze Detectors (JavaScript)
```javascript
const analysisRequest = {
  detector_type: "person",
  recompute_embeddings: false
};

const response = await fetch(`http://localhost:8000/detectors/case/${caseId}/detectors/analyze`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(analysisRequest)
});
```

### Update Settings (JavaScript)
```javascript
const settings = {
  face_thresholds: {
    high_confidence: 0.95,
    medium_confidence: 0.8,
    low_confidence: 0.65,
    minimum_match: 0.96
  }
};

const response = await fetch(`http://localhost:8000/detectors/case/${caseId}/detector-settings`, {
  method: 'PUT',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(settings)
});
```

---

## Notes

1. **Image Validation**: Strict validation ensures only valid face/object images are accepted
2. **Embedding Processing**: Background processing for embedding generation and matching
3. **Configurable Thresholds**: Flexible similarity thresholds for different use cases
4. **Comprehensive Statistics**: Detailed match statistics and summaries
5. **File Management**: Automatic cleanup of images and associated data
6. **Background Tasks**: Asynchronous processing for resource-intensive operations
7. **Security**: User-specific access control and file path sanitization
8. **Performance**: Optimized queries with limits and appropriate indexing
