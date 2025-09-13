from abc import ABC, abstractmethod
from typing import Tuple

class CalorieEstimator(ABC):
    """Base class for calorie estimation from images"""
    
    @abstractmethod
    def estimate_calories(self, image_path: str) -> Tuple[float, str]:
        """
        Estimate calories from an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (estimated_calories, food_description)
        """
        pass
