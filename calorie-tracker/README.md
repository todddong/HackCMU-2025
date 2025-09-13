# Calorie Tracker with Enhanced AI Integration

A MyFitnessPal-style calorie tracking web application with **advanced AI-powered calorie estimation** using OpenAI's GPT-4o-mini Vision API combined with USDA nutritional database for maximum accuracy and cost efficiency.

## Features

- 📱 **Modern UI**: Clean, responsive interface inspired by MyFitnessPal
- 📊 **Real-time Stats**: Track total calories, meal count, and averages
- 📷 **Photo Upload**: Upload meal photos for AI-powered calorie estimation
- 🤖 **Enhanced AI System**: GPT-4o-mini Vision + USDA nutritional database for accurate calorie estimation
- 💾 **Persistent Storage**: SQLite database to store meal history
- 🗑️ **Meal Management**: Add, view, and delete meals
- 🔄 **Fallback System**: Automatically falls back to mock estimator if OpenAI API is unavailable

## Project Structure

```
calorie-tracker/
├── backend/
│   ├── app.py                 # FastAPI main application
│   ├── database.py           # Database configuration
│   ├── models.py             # SQLAlchemy models
│   ├── schemas.py            # Pydantic schemas
│   ├── requirements.txt      # Python dependencies
│   ├── env_example.txt       # Environment variables example
│   └── calorie_estimator/    # Calorie estimation system
│       ├── __init__.py
│       ├── base.py           # Abstract base class
│       ├── mock_estimator.py # Mock implementation (fallback)
│       ├── openai_estimator.py # Basic OpenAI GPT-4o-mini implementation
│       └── enhanced_estimator.py # Enhanced AI + USDA data integration
└── frontend/
    └── index.html            # Single-page application
```

## Quick Start

### 1. Install Dependencies

```bash
cd calorie-tracker/backend
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key (Optional but Recommended)

To use the enhanced AI calorie estimation, you need an OpenAI API key:

**Easy Setup:**
```bash
cd calorie-tracker/backend
python setup_api_key.py
```

**Manual Setup:**
1. Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
   
   Or create a `.env` file in the backend directory:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

**Note**: If no API key is provided, the app will automatically fall back to a mock estimator for development/testing.

### 3. Run the Application

```bash
cd calorie-tracker/backend
python app.py
```

The application will be available at `http://localhost:8000`

### 4. Using the App

1. **Add a Meal**: Fill in the meal name, description, and calories
2. **Upload Photo**: Click "Estimate Calories from Photo" to get AI estimates
3. **View Stats**: See your daily calorie totals and meal statistics
4. **Manage Meals**: View, edit, or delete your meal history

## How the Enhanced AI Calorie Estimation Works

The application uses a sophisticated two-stage approach combining OpenAI's GPT-4o-mini Vision API with USDA nutritional database:

### Stage 1: Food Identification (OpenAI GPT-4o-mini)
1. **Photo Analysis**: The AI examines the uploaded food image
2. **Food Identification**: Identifies all food items visible in the image
3. **Detailed Recognition**: Recognizes specific foods (e.g., "grilled chicken breast", "white rice", "steamed broccoli")

### Stage 2: Nutritional Data Lookup (USDA Database)
4. **Database Search**: Searches USDA FoodData Central for each identified food item
5. **Accurate Nutrition**: Retrieves precise nutritional information including:
   - Exact calorie counts
   - Protein, carbohydrate, and fat content
   - Serving size information
   - Brand-specific data when available

### Stage 3: Intelligent Estimation
6. **Portion Analysis**: AI estimates portion sizes relative to standard servings
7. **Cooking Method Consideration**: Accounts for preparation methods (fried, grilled, raw, etc.)
8. **Hidden Ingredients**: Identifies oils, sauces, and other additions
9. **Comprehensive Response**: Provides detailed breakdown with accurate calorie totals

## API Endpoints

- `GET /` - Serve the main application
- `POST /api/meals/` - Create a new meal
- `GET /api/meals/` - Get all meals
- `DELETE /api/meals/{id}` - Delete a meal
- `POST /api/estimate-calories/` - Estimate calories from image
- `GET /api/stats/` - Get calorie statistics

## Technology Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **AI**: OpenAI GPT-4 Vision API for real calorie estimation
- **Styling**: Modern CSS with gradients and animations

## Development Notes

- The application automatically detects if OpenAI API key is available
- Falls back to mock estimator if API key is missing (for development)
- Photos are stored in the `uploads/` directory
- The database is automatically created on first run
- Real AI integration provides accurate calorie estimates from food photos

## Cost Considerations

- **GPT-4o-mini**: Extremely cost-effective at ~$0.00015 per image analysis
- **USDA Database**: Free to use (no API costs)
- **Total Cost**: Less than $0.001 per food photo analysis
- **Mock Estimator**: Free and available for development/testing

## Future Enhancements

- User authentication and profiles
- Meal categories and nutrition tracking
- Data export and reporting
- Mobile app development
- Batch photo processing
- Nutrition breakdown (protein, carbs, fats)
