# Alert API Specification

This document provides a comprehensive specification for all alert-related API endpoints.

## Base Information
- **Base URL**: `http://localhost:8000/alerts`
- **Authentication**: Bearer Token (JWT) required for all endpoints
- **Content-Type**: `application/json`

---

## Endpoints

### 1. Create Alert

**Endpoint**: `POST /`

**Description**: Creates a new alert for the authenticated user

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
  "toxicity_score": "number",
  "risk_level": ["string"],
  "sentiment_aspect": ["string"],
  "emotion": ["string"],
  "language": ["string"],
  "interaction_type": ["string"],
  "top_topic": ["string"],
  "description": "string",
  "entities": ["string"]
}
```

**Field Details**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | ✅ Mandatory | Alert name/identifier |
| toxicity_score | float | ❌ Optional | Toxicity score (0.0 to 1.0) |
| risk_level | List[string] | ❌ Optional | Array of risk level indicators |
| sentiment_aspect | List[string] | ❌ Optional | Array of sentiment aspects |
| emotion | List[string] | ❌ Optional | Array of detected emotions |
| language | List[string] | ❌ Optional | Array of detected languages |
| interaction_type | List[string] | ❌ Optional | Array of interaction types |
| top_topic | List[string] | ❌ Optional | Array of top topics |
| description | string | ❌ Optional | Alert description |
| entities | List[string] | ❌ Optional | Array of detected entities |

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

### 2. Get All User Alerts

**Endpoint**: `GET /`

**Description**: Retrieves all alerts for the authenticated user with user information populated

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
    "name": "High Risk Alert",
    "toxicity_score": 0.85,
    "risk_level": ["high", "critical"],
    "sentiment_aspect": ["negative", "aggressive"],
    "emotion": ["anger", "frustration"],
    "language": ["en", "ar"],
    "interaction_type": ["chat", "email"],
    "top_topic": ["security", "threat"],
    "description": "Alert for high-risk content detection",
    "entities": ["person", "organization"],
    "user": {
      "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
      "name": "John Doe",
      "email": "john.doe@example.com"
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

**Additional Details**:
- Uses MongoDB aggregation pipeline to join with Users collection
- Returns alerts with populated user information
- All ObjectIds are converted to strings in the response

---

### 3. Get Alert by ID

**Endpoint**: `GET /{alert_id}`

**Description**: Retrieves a specific alert by its ID for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| alert_id | string | ✅ Mandatory | MongoDB ObjectId of the alert |

**Query Parameters**: None

**Request Body**: None

**Response**:

**Success (200)**:
```json
{
  "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "user_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "name": "Security Alert",
  "toxicity_score": 0.75,
  "risk_level": ["medium", "high"],
  "sentiment_aspect": ["negative"],
  "emotion": ["concern", "worry"],
  "language": ["en"],
  "interaction_type": ["message"],
  "top_topic": ["security"],
  "description": "Security-related alert",
  "entities": ["person", "location"],
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
  "detail": "Alert not found or unauthorized"
}
```

**Additional Details**:
- Only returns alerts owned by the authenticated user
- ObjectIds are converted to strings in the response

---

### 4. Update Alert

**Endpoint**: `PUT /{alert_id}`

**Description**: Updates an existing alert for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| alert_id | string | ✅ Mandatory | MongoDB ObjectId of the alert to update |

**Request Body**:
```json
{
  "name": "string",
  "toxicity_score": "number",
  "risk_level": ["string"],
  "sentiment_aspect": ["string"],
  "emotion": ["string"],
  "language": ["string"],
  "interaction_type": ["string"],
  "top_topic": ["string"],
  "description": "string",
  "entities": ["string"]
}
```

**Field Details**: Same as Create Alert (all fields optional for updates)

**Response**:

**Success (200)**:
```json
{
  "message": "Alert updated successfully"
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
  "detail": "Alert not found or unauthorized"
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
- Only updates fields that are not null in the request
- Only allows updating alerts owned by the authenticated user

---

### 5. Delete Alert

**Endpoint**: `DELETE /{alert_id}`

**Description**: Deletes a specific alert for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| alert_id | string | ✅ Mandatory | MongoDB ObjectId of the alert to delete |

**Query Parameters**: None

**Request Body**: None

**Response**:

**Success (200)**:
```json
{
  "message": "Alert deleted successfully"
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
  "detail": "Alert not found or unauthorized"
}
```

**Additional Details**:
- Only allows deleting alerts owned by the authenticated user
- Permanent deletion from the database

---

## Data Models

### AlertModel Schema
```typescript
{
  name: string;                           // Required - Alert name/identifier
  toxicity_score?: number | null;         // Optional - Toxicity score (0.0-1.0)
  risk_level?: string[] | null;           // Optional - Risk level indicators
  sentiment_aspect?: string[] | null;     // Optional - Sentiment aspects
  emotion?: string[] | null;              // Optional - Detected emotions
  language?: string[] | null;             // Optional - Detected languages
  interaction_type?: string[] | null;     // Optional - Interaction types
  top_topic?: string[] | null;            // Optional - Top topics
  description?: string | null;            // Optional - Alert description
  entities?: string[] | null;             // Optional - Detected entities
}
```

### Alert Response Schema (Individual)
```typescript
{
  _id: string;                            // MongoDB ObjectId as string
  user_id: string;                        // User's MongoDB ObjectId as string
  name: string;                           // Alert name
  toxicity_score?: number | null;         // Toxicity score
  risk_level?: string[] | null;           // Risk levels
  sentiment_aspect?: string[] | null;     // Sentiment aspects
  emotion?: string[] | null;              // Emotions
  language?: string[] | null;             // Languages
  interaction_type?: string[] | null;     // Interaction types
  top_topic?: string[] | null;            // Topics
  description?: string | null;            // Description
  entities?: string[] | null;             // Entities
  created_at: string;                     // ISO datetime string
}
```

### Alert Response Schema (With User Info)
```typescript
{
  _id: string;                            // Alert MongoDB ObjectId as string
  name: string;                           // Alert name
  toxicity_score?: number | null;         // Toxicity score
  risk_level?: string[] | null;           // Risk levels
  sentiment_aspect?: string[] | null;     // Sentiment aspects
  emotion?: string[] | null;              // Emotions
  language?: string[] | null;             // Languages
  interaction_type?: string[] | null;     // Interaction types
  top_topic?: string[] | null;            // Topics
  description?: string | null;            // Description
  entities?: string[] | null;             // Entities
  user: {                                 // Populated user information
    _id: string;                          // User's MongoDB ObjectId as string
    name: string;                         // User's name
    email: string;                        // User's email
  };
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
- **404**: Not Found (alert not found or unauthorized)
- **422**: Unprocessable Entity (validation errors)

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

### Create Alert (cURL)
```bash
curl -X POST "http://localhost:8000/alerts/" \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Security Alert",
    "toxicity_score": 0.8,
    "risk_level": ["high"],
    "sentiment_aspect": ["negative"],
    "emotion": ["anger"],
    "language": ["en"],
    "interaction_type": ["chat"],
    "top_topic": ["security"],
    "description": "High-risk security alert",
    "entities": ["person", "organization"]
  }'
```

### Get All Alerts (JavaScript)
```javascript
const response = await fetch('http://localhost:8000/alerts/', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
const alerts = await response.json();
```

### Update Alert (JavaScript)
```javascript
const alertData = {
  name: "Updated Alert",
  description: "Updated description",
  risk_level: ["medium"]
};

const response = await fetch(`http://localhost:8000/alerts/${alertId}`, {
  method: 'PUT',
  headers: {
    'Authorization': `Bearer ${token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(alertData)
});
```

### Delete Alert (cURL)
```bash
curl -X DELETE "http://localhost:8000/alerts/64f1a2b3c4d5e6f7g8h9i0j1" \
  -H "Authorization: Bearer your_jwt_token"
```

---

## Advanced Features

### MongoDB Aggregation Pipeline
The GET `/` endpoint uses a sophisticated aggregation pipeline that:
1. **Matches** alerts by user_id
2. **Joins** with Users collection to populate user information
3. **Projects** specific fields and converts ObjectIds to strings
4. **Returns** enriched alert data with user details

### Partial Updates
The PUT endpoint supports partial updates:
- Only non-null fields in the request are updated
- Existing fields not included in the request remain unchanged
- Uses MongoDB's `$set` operator for efficient updates

### User Isolation
All operations are user-specific:
- Alerts are filtered by the authenticated user's ID
- Users can only access, modify, or delete their own alerts
- Cross-user access is prevented at the database query level

---

## Notes

1. **User Isolation**: All alerts are user-specific and isolated by user_id
2. **Automatic Timestamps**: Created timestamps are automatically added in UTC ISO format
3. **ObjectId Conversion**: MongoDB ObjectIds are converted to strings in all responses
4. **Flexible Arrays**: Most fields accept arrays of strings for multi-value support
5. **Toxicity Scoring**: Toxicity scores are typically float values between 0.0 and 1.0
6. **Aggregation Joins**: The list endpoint includes user information via MongoDB aggregation
7. **Partial Updates**: Update operations only modify non-null fields from the request
8. **Authorization**: All operations include ownership verification for security
