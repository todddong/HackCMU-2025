import random
from typing import Tuple
from .base import CalorieEstimator

class MockCalorieEstimator(CalorieEstimator):
    """Mock calorie estimator for development/testing"""
    
    # Sample food items with realistic calorie ranges
    FOOD_ITEMS = [
        ("Apple", 80, 120),
        ("Banana", 90, 110),
        ("Chicken Breast", 200, 300),
        ("Pizza Slice", 250, 400),
        ("Salad", 50, 150),
        ("Sandwich", 300, 500),
        ("Pasta", 200, 400),
        ("Rice Bowl", 300, 500),
        ("Burger", 400, 700),
        ("Soup", 100, 250),
        ("Eggs", 140, 200),
        ("Fish", 150, 250),
        ("Steak", 300, 500),
        ("Vegetables", 30, 80),
        ("Bread", 80, 120),
        ("Cheese", 100, 200),
        ("Nuts", 150, 300),
        ("Yogurt", 100, 200),
        ("Smoothie", 200, 400),
        ("Cereal", 150, 300)
    ]
    
    def estimate_calories(self, image_path: str) -> Tuple[float, str]:
        """
        Mock implementation that returns random food items and calories
        
        Args:
            image_path: Path to the image file (not used in mock)
            
        Returns:
            Tuple of (estimated_calories, food_description)
        """
        # Randomly select a food item
        food_name, min_cal, max_cal = random.choice(self.FOOD_ITEMS)
        
        # Generate random calories within the range
        estimated_calories = random.uniform(min_cal, max_cal)
        
        # Add some variation to the description
        descriptions = [
            f"Looks like {food_name.lower()}",
            f"Appears to be {food_name.lower()}",
            f"Detected {food_name.lower()}",
            f"Seems to be {food_name.lower()}"
        ]
        
        description = random.choice(descriptions)
        
        return round(estimated_calories, 1), description
