from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Authentication Schemas
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Meal Schemas
class MealBase(BaseModel):
    name: str
    calories: float
    description: Optional[str] = None

class MealCreate(MealBase):
    pass

class Meal(MealBase):
    id: int
    photo_path: Optional[str] = None
    created_at: datetime
    user_id: Optional[int] = None

    class Config:
        from_attributes = True

class MealWithPhoto(Meal):
    photo_url: Optional[str] = None
