from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List
import os
import shutil
from datetime import datetime

import models, schemas, database
from calorie_estimator.enhanced_estimator import EnhancedCalorieEstimator

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="Calorie Tracker API", version="1.0.0")

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize calorie estimator
try:
    calorie_estimator = EnhancedCalorieEstimator()
    print("✅ Enhanced AI estimator initialized (OpenAI GPT-4o-mini + USDA data)")
except ValueError as e:
    print(f"Warning: {e}")
    print("Falling back to mock estimator. Set OPENAI_API_KEY environment variable to use real AI analysis.")
    from calorie_estimator.mock_estimator import MockCalorieEstimator
    calorie_estimator = MockCalorieEstimator()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    import os
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_path)

@app.post("/api/meals/", response_model=schemas.Meal)
async def create_meal(
    name: str = Form(...),
    calories: float = Form(...),
    description: str = Form(None),
    photo: UploadFile = File(None),
    db: Session = Depends(database.get_db)
):
    """Create a new meal entry"""
    photo_path = None
    
    if photo:
        # Save uploaded photo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{photo.filename}"
        photo_path = os.path.join(UPLOAD_DIR, filename)
        
        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
    
    # Create meal in database
    db_meal = models.Meal(
        name=name,
        calories=calories,
        photo_path=photo_path,
        description=description
    )
    db.add(db_meal)
    db.commit()
    db.refresh(db_meal)
    
    return db_meal

@app.post("/api/estimate-calories/")
async def estimate_calories(photo: UploadFile = File(...)):
    """Estimate calories from uploaded image"""
    if not photo.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save temporary file
    temp_path = f"temp_{photo.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)
    
    try:
        # Estimate calories
        estimated_calories, description = calorie_estimator.estimate_calories(temp_path)
        
        return {
            "estimated_calories": estimated_calories,
            "description": description
        }
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/meals/", response_model=List[schemas.MealWithPhoto])
async def get_meals(db: Session = Depends(database.get_db)):
    """Get all meals"""
    meals = db.query(models.Meal).order_by(models.Meal.created_at.desc()).all()
    
    result = []
    for meal in meals:
        meal_dict = {
            "id": meal.id,
            "name": meal.name,
            "calories": meal.calories,
            "description": meal.description,
            "photo_path": meal.photo_path,
            "created_at": meal.created_at,
            "photo_url": f"/uploads/{os.path.basename(meal.photo_path)}" if meal.photo_path else None
        }
        result.append(meal_dict)
    
    return result

@app.delete("/api/meals/{meal_id}")
async def delete_meal(meal_id: int, db: Session = Depends(database.get_db)):
    """Delete a meal"""
    meal = db.query(models.Meal).filter(models.Meal.id == meal_id).first()
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    
    # Delete photo file if exists
    if meal.photo_path and os.path.exists(meal.photo_path):
        os.remove(meal.photo_path)
    
    db.delete(meal)
    db.commit()
    
    return {"message": "Meal deleted successfully"}

@app.get("/api/stats/")
async def get_stats(db: Session = Depends(database.get_db)):
    """Get calorie statistics"""
    meals = db.query(models.Meal).all()
    
    if not meals:
        return {
            "total_calories": 0,
            "total_meals": 0,
            "average_calories": 0
        }
    
    total_calories = sum(meal.calories for meal in meals)
    total_meals = len(meals)
    average_calories = total_calories / total_meals
    
    return {
        "total_calories": round(total_calories, 1),
        "total_meals": total_meals,
        "average_calories": round(average_calories, 1)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
