# Processing Profile API Specification

This document provides a comprehensive specification for all processing profile and model repository-related API endpoints.

## Base Information
- **Base URL**: `http://localhost:8000/models-profile`
- **Authentication**: Bearer Token (JWT) required for all endpoints
- **Content-Type**: `application/json`

---

## Endpoints

### 1. Create Processing Profile

**Endpoint**: `POST /`

**Description**: Creates a new processing profile for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "name": "string",
  "description": "string",
  "classifier": {
    "model": "string",
    "enabled": "boolean"
  },
  "emotion": {
    "model": "string",
    "enabled": "boolean"
  },
  "embeddings": {
    "model": "string",
    "enabled": "boolean",
    "size": "integer"
  },
  "toxic": {
    "model": "string",
    "enabled": "boolean"
  }
}
```

**Field Details**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | ✅ Mandatory | Profile name/identifier |
| description | string | ❌ Optional | Profile description (can be null) |
| classifier | Dict[str, str\|bool] | ✅ Mandatory | Classifier configuration object |
| emotion | Dict[str, str\|bool] | ✅ Mandatory | Emotion analysis configuration object |
| embeddings | Dict[str, str\|bool\|int] | ✅ Mandatory | Embeddings configuration object |
| toxic | Dict[str, str\|bool] | ✅ Mandatory | Toxicity detection configuration object |

**Response**:

**Success (200)**:
```json
{
  "_id": "64f1a2b3c4d5e6f7g8h9i0j1"
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Error (422)**:
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

### 2. Get All Processing Profiles

**Endpoint**: `GET /`

**Description**: Retrieves all processing profiles for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Query Parameters**: None

**Path Parameters**: None

**Request Body**: None

**Response**:

**Success (200)**:
```json
[
  {
    "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "user_id": "64f1a2b3c4d5e6f7g8h9i0j1",
    "name": "Default Profile",
    "description": "Standard processing configuration",
    "classifier": {
      "model": "xlm-roberta-large-xnli",
      "enabled": true
    },
    "emotion": {
      "model": "twitter-xlm-roberta-base-sentiment",
      "enabled": true
    },
    "embeddings": {
      "model": "minilm-l12-v2",
      "enabled": true,
      "size": 384
    },
    "toxic": {
      "model": "akhooli-xlm-large-arabic-toxic",
      "enabled": false
    },
    "created_at": "2024-12-01T12:00:00.000000+00:00"
  }
]
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Error (500)**:
```json
{
  "detail": "Some thing went wrong"
}
```

---

### 3. Get Processing Profile by ID

**Endpoint**: `GET /{profile_id}`

**Description**: Retrieves a specific processing profile by its ID for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| profile_id | string | ✅ Mandatory | MongoDB ObjectId of the profile |

**Query Parameters**: None

**Request Body**: None

**Response**:

**Success (200)**:
```json
{
  "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "user_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "name": "Custom Profile",
  "description": "Custom processing configuration",
  "classifier": {
    "model": "xlm-roberta-large-xnli",
    "enabled": true
  },
  "emotion": {
    "model": "twitter-xlm-roberta-base-sentiment",
    "enabled": true
  },
  "embeddings": {
    "model": "minilm-l12-v2",
    "enabled": true,
    "size": 384
  },
  "toxic": {
    "model": "akhooli-xlm-large-arabic-toxic",
    "enabled": false
  },
  "created_at": "2024-12-01T12:00:00.000000+00:00"
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Error (404)**:
```json
{
  "detail": "Profile not found"
}
```

---

### 4. Update Processing Profile

**Endpoint**: `PUT /{profile_id}`

**Description**: Updates an existing processing profile for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| profile_id | string | ✅ Mandatory | MongoDB ObjectId of the profile to update |

**Request Body**:
```json
{
  "name": "string",
  "description": "string",
  "classifier": {
    "model": "string",
    "enabled": "boolean"
  },
  "emotion": {
    "model": "string",
    "enabled": "boolean"
  },
  "embeddings": {
    "model": "string",
    "enabled": "boolean",
    "size": "integer"
  },
  "toxic": {
    "model": "string",
    "enabled": "boolean"
  }
}
```

**Field Details**: Same as Create Processing Profile

**Response**:

**Success (200)**:
```json
{
  "message": "Profile updated successfully"
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Error (404)**:
```json
{
  "detail": "Profile not found"
}
```

**Error (422)**:
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

### 5. Delete Processing Profile

**Endpoint**: `DELETE /{profile_id}`

**Description**: Deletes a specific processing profile for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| profile_id | string | ✅ Mandatory | MongoDB ObjectId of the profile to delete |

**Query Parameters**: None

**Request Body**: None

**Response**:

**Success (200)**:
```json
{
  "message": "Profile deleted successfully"
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Error (404)**:
```json
{
  "detail": "Profile not found"
}
```

---

### 6. Create Model Repository Entry

**Endpoint**: `POST /model-repository`

**Description**: Creates a new model entry in the model repository

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "model": "string",
  "name": "string",
  "type": "string",
  "is_local": "boolean",
  "base_type": "string",
  "embedding_size": "integer"
}
```

**Field Details**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| model | string | ✅ Mandatory | Model identifier/path |
| name | string | ✅ Mandatory | Human-readable model name |
| type | string | ✅ Mandatory | Model type specification |
| is_local | boolean | ✅ Mandatory | Whether model is stored locally |
| base_type | string | ✅ Mandatory | Base category of the model |
| embedding_size | integer | ❌ Optional | Size of embeddings (only for embedding models) |

**Response**:

**Success (200)**:
```json
{
  "_id": "64f1a2b3c4d5e6f7g8h9i0j1"
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Error (422)**:
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Additional Details**:
- If `base_type` is not "embeddings", the `embedding_size` field is automatically removed
- `created_at` timestamp is automatically added in UTC ISO format

---

### 7. Get Models Grouped by Base Type

**Endpoint**: `GET /model-repository/grouped`

**Description**: Retrieves all models from the repository grouped by their base type (excludes entity type models)

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Query Parameters**: None

**Path Parameters**: None

**Request Body**: None

**Response**:

**Success (200)**:
```json
{
  "classifier": [
    {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
      "model": "xlm-roberta-large-xnli",
      "name": "XLM-RoBERTa Large XNLI",
      "type": "classification",
      "is_local": true,
      "base_type": "classifier",
      "created_at": "2024-12-01T12:00:00.000000+00:00"
    }
  ],
  "emotion": [
    {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j2",
      "model": "twitter-xlm-roberta-base-sentiment",
      "name": "Twitter XLM-RoBERTa Base Sentiment",
      "type": "sentiment_analysis",
      "is_local": true,
      "base_type": "emotion",
      "created_at": "2024-12-01T12:00:00.000000+00:00"
    }
  ],
  "embeddings": [
    {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j3",
      "model": "minilm-l12-v2",
      "name": "MiniLM L12 v2",
      "type": "sentence_embedding",
      "is_local": true,
      "base_type": "embeddings",
      "embedding_size": 384,
      "created_at": "2024-12-01T12:00:00.000000+00:00"
    }
  ],
  "toxic": [
    {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j4",
      "model": "akhooli-xlm-large-arabic-toxic",
      "name": "Akhooli XLM Large Arabic Toxic",
      "type": "toxicity_detection",
      "is_local": true,
      "base_type": "toxic",
      "created_at": "2024-12-01T12:00:00.000000+00:00"
    }
  ]
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Error (500)**:
```json
{
  "detail": "Something went wrong"
}
```

**Additional Details**:
- Models with `base_type` of "entity" are excluded from results
- Models are grouped by their `base_type` field
- Each group contains an array of model objects

---

## Data Models

### ProcessingProfile Schema
```typescript
{
  name: string;                           // Required - Profile name
  description?: string | null;            // Optional - Profile description
  classifier: {                           // Required - Classifier configuration
    model?: string;                       // Model identifier
    enabled?: boolean;                    // Whether classifier is enabled
    [key: string]: string | boolean;      // Additional properties
  };
  emotion: {                              // Required - Emotion analysis configuration
    model?: string;                       // Model identifier
    enabled?: boolean;                    // Whether emotion analysis is enabled
    [key: string]: string | boolean;      // Additional properties
  };
  embeddings: {                           // Required - Embeddings configuration
    model?: string;                       // Model identifier
    enabled?: boolean;                    // Whether embeddings are enabled
    size?: number;                        // Embedding dimensions
    [key: string]: string | boolean | number; // Additional properties
  };
  toxic: {                                // Required - Toxicity detection configuration
    model?: string;                       // Model identifier
    enabled?: boolean;                    // Whether toxicity detection is enabled
    [key: string]: string | boolean;      // Additional properties
  };
}
```

### ModelRepository Schema
```typescript
{
  model: string;                          // Required - Model identifier/path
  name: string;                           // Required - Human-readable name
  type: string;                           // Required - Model type
  is_local: boolean;                      // Required - Local storage flag
  base_type: string;                      // Required - Base category
  embedding_size?: number;                // Optional - Embedding size (embeddings only)
  created_at?: string;                    // Auto-generated - ISO datetime
}
```

### Profile Response Schema
```typescript
{
  _id: string;                            // MongoDB ObjectId as string
  user_id: string;                        // User's MongoDB ObjectId as string
  name: string;                           // Profile name
  description?: string | null;            // Profile description
  classifier: object;                     // Classifier configuration
  emotion: object;                        // Emotion configuration
  embeddings: object;                     // Embeddings configuration
  toxic: object;                          // Toxicity configuration
  created_at: string;                     // ISO datetime string
}
```

---

## Authentication Details

### Required Authentication
All endpoints require JWT authentication:
- Include Bearer token in Authorization header
- Token must be valid and not expired
- User ID is extracted from JWT payload for user-specific operations

### Token Usage
```
Authorization: Bearer <your_jwt_token>
```

---

## Error Handling

### Common HTTP Status Codes
- **200**: Success
- **401**: Unauthorized (invalid/expired token)
- **404**: Not Found (profile not found)
- **422**: Unprocessable Entity (validation errors)
- **500**: Internal Server Error

### Error Response Format
All errors follow this structure:
```json
{
  "detail": "Error message description"
}
```

### Validation Error Format
```json
{
  "detail": [
    {
      "loc": ["body", "field_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Usage Examples

### Create Processing Profile (cURL)
```bash
curl -X POST "http://localhost:8000/models-profile/" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Profile",
    "description": "Custom processing profile",
    "classifier": {"model": "xlm-roberta", "enabled": true},
    "emotion": {"model": "twitter-sentiment", "enabled": true},
    "embeddings": {"model": "minilm", "enabled": true, "size": 384},
    "toxic": {"model": "akhooli-toxic", "enabled": false}
  }'
```

### Get All Profiles (JavaScript)
```javascript
const response = await fetch('http://localhost:8000/models-profile/', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
const profiles = await response.json();
```

### Create Model Repository Entry (JavaScript)
```javascript
const modelData = {
  model: "new-model-v1",
  name: "New Model Version 1",
  type: "classification",
  is_local: true,
  base_type: "classifier"
};

const response = await fetch('http://localhost:8000/models-profile/model-repository', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(modelData)
});
```

---

## Notes

1. **User Isolation**: All profiles are user-specific and isolated by user_id
2. **Automatic Timestamps**: Created timestamps are automatically added in UTC ISO format
3. **ObjectId Conversion**: MongoDB ObjectIds are converted to strings in responses
4. **Flexible Configuration**: Processing profile configurations use flexible Dict types for extensibility
5. **Model Filtering**: Model repository grouped endpoint excludes entity-type models
6. **Embedding Size Handling**: Embedding size is automatically removed for non-embedding models
7. **Error Handling**: Comprehensive error handling with specific status codes and messages
8. **Validation**: Pydantic models provide automatic request validation
