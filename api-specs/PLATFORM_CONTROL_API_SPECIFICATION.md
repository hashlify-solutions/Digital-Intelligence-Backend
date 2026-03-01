# Platform Control API Specification

This document provides a comprehensive specification for all platform control-related API endpoints.

## Base Information
- **Base URL**: `http://localhost:8000/platform-control`
- **Authentication**: Bearer Token (JWT) required for all endpoints
- **Content-Type**: `multipart/form-data` (for file uploads) or `application/json`

---

## Endpoints

### 1. Update Platform Details

**Endpoint**: `POST /update`

**Description**: Updates platform details including name and logo. Creates or updates the platform configuration for the authenticated user.

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: multipart/form-data
```

**Request Type**: `multipart/form-data`

**Form Data Fields**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| name | string | ✅ Mandatory | Platform name/title | Any string value |
| logo | file (UploadFile) | ✅ Mandatory | Platform logo image file | Must be an image file (image/*) |

**Request Example**:
```
POST /platform-control/update
Content-Type: multipart/form-data

name: "My Digital Intelligence Platform"
logo: [image file - .jpg, .png, .gif, etc.]
```

**Response**:

**Success (200)**:
```json
{
  "status": "success",
  "message": "Platform details updated successfully",
  "data": {
    "name": "My Digital Intelligence Platform",
    "logo_path": "/logo/64f1a2b3c4d5e6f7g8h9i0j1/platform_logo_20241201120000.png"
  }
}
```

**Error (400)**:
```json
{
  "detail": "File must be an image"
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
  "detail": "Error updating platform details: [specific error message]"
}
```

**Additional Details**:
- Logo files are saved to user-specific directories: `/logo/{user_id}/`
- Filename format: `platform_logo_{timestamp}{extension}`
- Old logo files are automatically deleted when updating
- Supported image formats: Any format with `image/*` MIME type
- File path returned is relative and can be used directly by frontend

---

### 2. Get Platform Details

**Endpoint**: `GET /details`

**Description**: Retrieves the current platform configuration for the authenticated user

**Authentication**: ✅ Required (Bearer Token)

**Headers**:
```
Authorization: Bearer <jwt_token>
```

**Query Parameters**: None

**Path Parameters**: None

**Request Body**: None

**Response**:

**Success (200) - With Existing Configuration**:
```json
{
  "_id": "64f1a2b3c4d5e6f7g8h9i0j1",
  "userId": "64f1a2b3c4d5e6f7g8h9i0j1",
  "name": "My Digital Intelligence Platform",
  "logo": "/logo/64f1a2b3c4d5e6f7g8h9i0j1/platform_logo_20241201120000.png",
  "created_at": "2024-12-01T12:00:00.000Z",
  "updated_at": "2024-12-01T12:30:00.000Z"
}
```

**Success (200) - Default Configuration (No existing record)**:
```json
{
  "name": "Digital Intelligence Platform",
  "logo": null,
  "userId": "64f1a2b3c4d5e6f7g8h9i0j1"
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

**Additional Details**:
- If no platform configuration exists, returns default values
- Default platform name: "Digital Intelligence Platform"
- Default logo: `null`
- MongoDB ObjectIds are converted to strings in the response

---

## Data Models

### Platform Control Document Structure
```typescript
{
  _id?: string;                    // MongoDB ObjectId (only in responses)
  userId: string;                  // User's MongoDB ObjectId
  name: string;                    // Platform name/title
  logo: string;                    // Relative path to logo file
  created_at?: string;             // ISO datetime string (only for new records)
  updated_at?: string;             // ISO datetime string (updated on each change)
}
```

### Update Request Schema
```typescript
{
  name: string;                    // Required - Platform name
  logo: File;                      // Required - Image file (UploadFile)
}
```

### Success Response Schema
```typescript
{
  status: "success";               // Always "success" for successful operations
  message: string;                 // Success message
  data: {
    name: string;                  // Updated platform name
    logo_path: string;             // Path to uploaded logo file
  };
}
```

---

## File Handling

### Logo File Management
- **Storage Location**: `/logo/{user_id}/`
- **Filename Pattern**: `platform_logo_{timestamp}{extension}`
- **Timestamp Format**: `YYYYMMDDHHMMSS`
- **Supported Formats**: All image formats (validated by MIME type `image/*`)
- **File Cleanup**: Old logo files are automatically deleted when updating

### File Path Structure
```
/logo/
  └── {user_id}/
      └── platform_logo_{timestamp}.{extension}
```

**Example**:
```
/logo/64f1a2b3c4d5e6f7g8h9i0j1/platform_logo_20241201120000.png
```

---

## Authentication Details

### Required Authentication
Both endpoints require JWT authentication:
- Include Bearer token in Authorization header
- Token must be valid and not expired
- User ID is extracted from the JWT payload

### Token Usage
```
Authorization: Bearer <your_jwt_token>
```

---

## Error Handling

### Common HTTP Status Codes
- **200**: Success
- **400**: Bad Request (invalid file type, validation errors)
- **401**: Unauthorized (invalid/expired token)
- **500**: Internal Server Error (file system errors, database errors)

### Error Response Format
All errors follow this structure:
```json
{
  "detail": "Error message description"
}
```

### Specific Error Cases

#### File Validation Errors
- **Non-image file**: `"File must be an image"`
- **Missing file**: FastAPI validation error for required field

#### Authentication Errors
- **Invalid token**: `"Invalid token"`
- **Expired token**: `"Token expired"`
- **Missing token**: `"Not authenticated"`

#### Server Errors
- **File system errors**: `"Error updating platform details: [specific error]"`
- **Database errors**: `"Error updating platform details: [specific error]"`

---

## Usage Examples

### Update Platform Details (cURL)
```bash
curl -X POST "http://localhost:8000/platform-control/update" \
  -H "Authorization: Bearer your_jwt_token" \
  -F "name=My Custom Platform" \
  -F "logo=@/path/to/logo.png"
```

### Get Platform Details (cURL)
```bash
curl -X GET "http://localhost:8000/platform-control/details" \
  -H "Authorization: Bearer your_jwt_token"
```

### JavaScript/Fetch Example
```javascript
// Update platform details
const formData = new FormData();
formData.append('name', 'My Platform');
formData.append('logo', logoFile); // File object from input

const response = await fetch('http://localhost:8000/platform-control/update', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
  },
  body: formData
});

// Get platform details
const details = await fetch('http://localhost:8000/platform-control/details', {
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

---

## Notes

1. **User Isolation**: Each user has their own platform configuration and logo storage directory
2. **File Management**: The system automatically handles file cleanup when logos are updated
3. **Default Values**: If no configuration exists, the system returns sensible defaults
4. **Static File Serving**: Logo files are served as static files through the `/logo/` route
5. **Timestamp Naming**: Logo files include timestamps to prevent naming conflicts
6. **MIME Type Validation**: Only files with `image/*` MIME types are accepted
7. **MongoDB Integration**: Uses MongoDB for storing platform configuration data
8. **Path Format**: Logo paths are returned in a format ready for frontend consumption
