from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    name: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: str
    name: str
    email: EmailStr

class UserLogin(BaseModel):
    email: EmailStr
    password: str