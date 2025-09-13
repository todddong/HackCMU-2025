# Calorie Tracker with OpenAI GPT-4 Vision

A MyFitnessPal-style calorie tracking web application with **real AI-powered calorie estimation** using OpenAI's GPT-4 Vision API to analyze food photos.

## Features

- 📱 **Modern UI**: Clean, responsive interface inspired by MyFitnessPal
- 📊 **Real-time Stats**: Track total calories, meal count, and averages
- 📷 **Photo Upload**: Upload meal photos for AI-powered calorie estimation
- 🤖 **OpenAI GPT-4 Vision**: Real AI system that analyzes food images and estimates calories
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
│       └── openai_estimator.py # OpenAI GPT-4 Vision implementation
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

To use the real AI calorie estimation, you need an OpenAI API key:

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

## How the AI Calorie Estimation Works

The application uses OpenAI's GPT-4 Vision API to analyze food photos and provide accurate calorie estimates:

1. **Photo Analysis**: The AI examines the uploaded food image
2. **Food Identification**: Identifies all food items visible in the image
3. **Portion Estimation**: Estimates portion sizes relative to common serving sizes
4. **Calorie Calculation**: Calculates approximate total calories considering:
   - Different food types (proteins, carbs, fats, vegetables)
   - Cooking methods (fried, grilled, raw, etc.)
   - Hidden ingredients (oils, sauces, etc.)
5. **Detailed Response**: Provides both calorie count and food description

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

- OpenAI GPT-4 Vision API charges per image analyzed
- Typical cost: ~$0.01-0.02 per food photo analysis
- Mock estimator is free and available for development/testing

## Future Enhancements

- User authentication and profiles
- Meal categories and nutrition tracking
- Data export and reporting
- Mobile app development
- Batch photo processing
- Nutrition breakdown (protein, carbs, fats)
