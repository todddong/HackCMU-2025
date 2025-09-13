from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MealBase(BaseModel):
    name: str
    calories: float
    protein: Optional[float] = 0.0
    carbs: Optional[float] = 0.0
    fat: Optional[float] = 0.0
    description: Optional[str] = None

class MealCreate(MealBase):
    pass

class Meal(MealBase):
    id: int
    photo_path: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True

class MealWithPhoto(Meal):
    photo_url: Optional[str] = None
