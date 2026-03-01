# User API Specification

This document provides a comprehensive specification for all user-related API endpoints.

## Base Information
- **Base URL**: `http://localhost:8000/users` (assuming router is mounted with this prefix)
- **Authentication**: Bearer Token (JWT) required for protected endpoints
- **Content-Type**: `application/json`

---

## Endpoints

### 1. User Signup

**Endpoint**: `POST /signup`

**Description**: Creates a new user account

**Authentication**: Not required

**Request Body**:
```json
{
  "name": "string",
  "email": "string (email format)",
  "password": "string"
}
```

**Field Details**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| name | string | ✅ Mandatory | User's full name |
| email | string (EmailStr) | ✅ Mandatory | Valid email address |
| password | string | ✅ Mandatory | User's password |

**Response**:

**Success (200)**:
```json
{
  "id": "string",
  "name": "string",
  "email": "string"
}
```

**Error (400)**:
```json
{
  "detail": "Email already registered"
}
```

---

### 2. User Login

**Endpoint**: `POST /login`

**Description**: Authenticates a user and returns access and refresh tokens

**Authentication**: Not required

**Request Body**:
```json
{
  "email": "string (email format)",
  "password": "string"
}
```

**Field Details**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| email | string (EmailStr) | ✅ Mandatory | User's email address |
| password | string | ✅ Mandatory | User's password |

**Response**:

**Success (200)**:
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "refresh_token": "string"
}
```

**Error (400)**:
```json
{
  "detail": "Invalid credentials"
}
```

---

### 3. Get User by Token

**Endpoint**: `GET /get-by-token`

**Description**: Retrieves the current user's information based on the provided JWT token

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
  "id": "string",
  "name": "string",
  "email": "string"
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
  "detail": "User not found"
}
```

---

### 4. Get All Users

**Endpoint**: `GET /`

**Description**: Retrieves a list of all users (limited to 100 users)

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
    "id": "string",
    "name": "string",
    "email": "string"
  }
]
```

**Error (401)**:
```json
{
  "detail": "Invalid token" | "Token expired"
}
```

---

### 5. Refresh Token

**Endpoint**: `POST /refresh-token`

**Description**: Generates new access and refresh tokens using a valid refresh token

**Authentication**: Not required (but requires valid refresh token in body)

**Request Body**:
```json
{
  "token": "string"
}
```

**Field Details**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| token | string | ✅ Mandatory | Valid refresh token |

**Response**:

**Success (200)**:
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "refresh_token": "string"
}
```

**Error (401)**:
```json
{
  "detail": "Invalid token"
}
```

---

## Authentication Details

### JWT Token Structure
- **Access Token Expiry**: 1440 minutes (24 hours)
- **Refresh Token Expiry**: 7 days
- **Token Type**: Bearer
- **Algorithm**: Configurable (from settings)

### Protected Endpoints
The following endpoints require authentication:
- `GET /get-by-token`
- `GET /`

### Token Usage
Include the JWT token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

---

## Error Handling

### Common HTTP Status Codes
- **200**: Success
- **400**: Bad Request (validation errors, duplicate email, invalid credentials)
- **401**: Unauthorized (invalid/expired token)
- **404**: Not Found (user not found)

### Error Response Format
All errors follow this structure:
```json
{
  "detail": "Error message description"
}
```

---

## Data Types

### UserCreate Schema
```typescript
{
  name: string;           // Required
  email: string;          // Required, must be valid email format
  password: string;       // Required
}
```

### UserOut Schema
```typescript
{
  id: string;            // MongoDB ObjectId as string
  name: string;          // User's full name
  email: string;         // User's email address
}
```

### UserLogin Schema
```typescript
{
  email: string;         // Required, must be valid email format
  password: string;      // Required
}
```

---

## Notes

1. **Password Security**: Passwords are hashed using bcrypt before storage
2. **Email Validation**: Email fields use Pydantic's EmailStr for validation
3. **Token Refresh**: The refresh token endpoint generates both new access and refresh tokens
4. **User Limit**: The get all users endpoint is limited to 100 users
5. **Password Exclusion**: User passwords are never returned in API responses
6. **MongoDB Integration**: User IDs are MongoDB ObjectIds converted to strings
