# Calorie Tracker MVP

A MyFitnessPal-style calorie tracking web application with AI-powered calorie estimation from food photos.

## Features

- 📱 **Modern UI**: Clean, responsive interface inspired by MyFitnessPal
- 📊 **Real-time Stats**: Track total calories, meal count, and averages
- 📷 **Photo Upload**: Upload meal photos for calorie estimation
- 🤖 **AI Estimation**: Mock AI system that estimates calories from food images
- 💾 **Persistent Storage**: SQLite database to store meal history
- 🗑️ **Meal Management**: Add, view, and delete meals

## Project Structure

```
calorie-tracker/
├── backend/
│   ├── app.py                 # FastAPI main application
│   ├── database.py           # Database configuration
│   ├── models.py             # SQLAlchemy models
│   ├── schemas.py            # Pydantic schemas
│   ├── requirements.txt      # Python dependencies
│   └── calorie_estimator/    # Calorie estimation system
│       ├── __init__.py
│       ├── base.py           # Abstract base class
│       └── mock_estimator.py # Mock implementation
└── frontend/
    └── index.html            # Single-page application
```

## Quick Start

### 1. Install Dependencies

```bash
cd calorie-tracker/backend
pip install -r requirements.txt
```

### 2. Run the Application

```bash
cd calorie-tracker/backend
python app.py
```

The application will be available at `http://localhost:8000`

### 3. Using the App

1. **Add a Meal**: Fill in the meal name, description, and calories
2. **Upload Photo**: Click "Estimate Calories from Photo" to get AI estimates
3. **View Stats**: See your daily calorie totals and meal statistics
4. **Manage Meals**: View, edit, or delete your meal history

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
- **AI**: Mock calorie estimator (easily replaceable with real AI)
- **Styling**: Modern CSS with gradients and animations

## Development Notes

- The calorie estimator is currently a mock implementation that returns random food items and calorie estimates
- Photos are stored in the `uploads/` directory
- The database is automatically created on first run
- The application is designed to be easily extensible for real AI integration

## Future Enhancements

- Real AI integration for calorie estimation
- User authentication and profiles
- Meal categories and nutrition tracking
- Data export and reporting
- Mobile app development
