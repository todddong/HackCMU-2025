from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from datetime import datetime, timedelta

import models, schemas, database, auth
from calorie_estimator.mock_estimator import MockCalorieEstimator
from calorie_estimator.enhanced_estimator import EnhancedCalorieEstimator
from usda_service import usda_service
from gpt_search_wrapper import gpt_search_wrapper

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="nutriAI API", version="1.0.0")

# Create uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Initialize calorie estimator with fallback
try:
    # Try to use enhanced AI estimator if OpenAI API key is available
    calorie_estimator = EnhancedCalorieEstimator()
    print("✅ Enhanced AI calorie estimator initialized successfully")
except Exception as e:
    # Fallback to mock estimator if OpenAI API key is not available
    calorie_estimator = MockCalorieEstimator()
    print(f"⚠️  Using mock calorie estimator (Enhanced AI not available: {e})")

# Security
security = HTTPBearer()

# ==================== AUTHENTICATION ROUTES ====================

@app.post("/api/auth/register", response_model=schemas.User)
async def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    """Register a new user"""
    # Check if username already exists
    existing_user = db.query(models.User).filter(models.User.username == user.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        password_hash=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@app.post("/api/auth/login", response_model=schemas.Token)
async def login(user_credentials: schemas.UserLogin, db: Session = Depends(database.get_db)):
    """Login and get access token"""
    user = auth.authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=schemas.User)
async def read_users_me(current_user: models.User = Depends(auth.get_current_user)):
    """Get current user information"""
    return current_user

@app.get("/api/auth/users", response_model=List[schemas.User])
async def read_users(db: Session = Depends(database.get_db)):
    """Get all users (for development)"""
    users = db.query(models.User).all()
    return users

# ==================== MAIN APPLICATION ROUTES ====================

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("../frontend/index.html")

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    """Serve the login page"""
    return FileResponse("../frontend/login.html")

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    """Serve the register page"""
    return FileResponse("../frontend/register.html")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the dashboard page"""
    return FileResponse("../frontend/dashboard.html")

# ==================== MEAL ROUTES (PROTECTED) ====================

@app.post("/api/meals/", response_model=schemas.Meal)
async def create_meal(
    name: str = Form(...),
    calories: float = Form(...),
    protein: float = Form(0.0),
    carbs: float = Form(0.0),
    fat: float = Form(0.0),
    description: str = Form(None),
    photo: UploadFile = File(None),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Create a new meal entry (requires authentication)"""
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
        protein=protein,
        carbs=carbs,
        fat=fat,
        photo_path=photo_path,
        description=description,
        user_id=current_user.id
    )
    db.add(db_meal)
    db.commit()
    db.refresh(db_meal)
    
    return db_meal

@app.get("/api/meals/", response_model=List[schemas.MealWithPhoto])
async def get_meals(
    db: Session = Depends(database.get_db),
    current_user: Optional[models.User] = Depends(auth.get_current_user_optional)
):
    """Get all meals (shows user's meals if authenticated, all meals if not)"""
    if current_user:
        # Show only current user's meals
        meals = db.query(models.Meal).filter(models.Meal.user_id == current_user.id).order_by(models.Meal.created_at.desc()).all()
    else:
        # Show all meals (for demo purposes)
        meals = db.query(models.Meal).order_by(models.Meal.created_at.desc()).all()
    
    result = []
    for meal in meals:
        meal_dict = {
            "id": meal.id,
            "name": meal.name,
            "calories": meal.calories,
            "protein": meal.protein,
            "carbs": meal.carbs,
            "fat": meal.fat,
            "description": meal.description,
            "photo_path": meal.photo_path,
            "created_at": meal.created_at,
            "user_id": meal.user_id,
            "photo_url": f"/uploads/{os.path.basename(meal.photo_path)}" if meal.photo_path else None
        }
        result.append(meal_dict)
    
    return result

@app.delete("/api/meals/{meal_id}")
async def delete_meal(
    meal_id: int, 
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(auth.get_current_user)
):
    """Delete a meal (requires authentication)"""
    meal = db.query(models.Meal).filter(models.Meal.id == meal_id).first()
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    
    # Check if user owns this meal
    if meal.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this meal")
    
    # Delete photo file if exists
    if meal.photo_path and os.path.exists(meal.photo_path):
        os.remove(meal.photo_path)
    
    db.delete(meal)
    db.commit()
    
    return {"message": "Meal deleted successfully"}

@app.get("/api/stats/")
async def get_stats(
    db: Session = Depends(database.get_db),
    current_user: Optional[models.User] = Depends(auth.get_current_user_optional)
):
    """Get calorie statistics"""
    if current_user:
        # Show only current user's stats
        meals = db.query(models.Meal).filter(models.Meal.user_id == current_user.id).all()
    else:
        # Show all meals stats (for demo purposes)
        meals = db.query(models.Meal).all()
    
    if not meals:
        return {
            "total_calories": 0,
            "total_protein": 0,
            "total_carbs": 0,
            "total_fat": 0,
            "total_meals": 0,
            "average_calories": 0
        }
    
    total_calories = sum(meal.calories for meal in meals)
    total_protein = sum(meal.protein for meal in meals)
    total_carbs = sum(meal.carbs for meal in meals)
    total_fat = sum(meal.fat for meal in meals)
    total_meals = len(meals)
    average_calories = total_calories / total_meals
    
    return {
        "total_calories": round(total_calories, 1),
        "total_protein": round(total_protein, 1),
        "total_carbs": round(total_carbs, 1),
        "total_fat": round(total_fat, 1),
        "total_meals": total_meals,
        "average_calories": round(average_calories, 1)
    }

# ==================== FOOD SEARCH ROUTES ====================

@app.get("/api/search-foods/")
async def search_foods(
    query: str = Query(..., description="Food search query"),
    page_size: int = Query(20, description="Number of results to return"),
    data_type: Optional[str] = Query(None, description="Filter by data type (Foundation, SR Legacy, Survey, Branded, Experimental)"),
    use_gpt: bool = Query(True, description="Use GPT-enhanced search (default: True)")
):
    """Search for foods using GPT-enhanced USDA FoodData Central API"""
    if not query or len(query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters long")
    
    try:
        if use_gpt:
            # Use GPT-enhanced search
            foods = await gpt_search_wrapper.search_foods(query.strip(), page_size, data_type)
            print(f"🤖 GPT-enhanced search for: {query}")
        else:
            # Use traditional USDA search
            foods = await usda_service.search_foods(query.strip(), page_size, data_type)
            print(f"🔍 Traditional USDA search for: {query}")
        
        return {
            "query": query,
            "results": foods,
            "total_results": len(foods),
            "search_type": "gpt_enhanced" if use_gpt else "traditional_usda"
        }
    except Exception as e:
        print(f"Error searching foods: {e}")
        raise HTTPException(status_code=500, detail="Failed to search foods")

@app.get("/api/food-details/{fdc_id}")
async def get_food_details(fdc_id: str):
    """Get detailed information for a specific food item"""
    try:
        food_details = await usda_service.get_food_details(fdc_id)
        if not food_details:
            raise HTTPException(status_code=404, detail="Food not found")
        return food_details
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting food details: {e}")
        raise HTTPException(status_code=500, detail="Failed to get food details")

# ==================== UTILITY ROUTES ====================

@app.post("/api/estimate-calories/")
async def estimate_calories(photo: UploadFile = File(...)):
    """Estimate calories from uploaded image (public endpoint)"""
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)