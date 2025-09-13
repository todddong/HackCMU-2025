import base64
import os
import asyncio
from typing import Tuple, List, Dict, Optional
from openai import OpenAI
from .base import CalorieEstimator
from ..usda_service import usda_service

class EnhancedCalorieEstimator(CalorieEstimator):
    """Enhanced calorie estimator combining OpenAI GPT-4o-mini Vision with USDA nutritional data"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the enhanced calorie estimator
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string for OpenAI API
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    async def estimate_calories(self, image_path: str) -> Tuple[float, str]:
        """
        Estimate calories from food image using OpenAI GPT-4o-mini Vision + USDA data
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (estimated_calories, food_description)
        """
        try:
            # Step 1: Use OpenAI to identify food items
            food_items = await self._identify_food_items(image_path)
            
            if not food_items:
                return 300.0, "Unable to identify food items in the image"
            
            # Step 2: Get nutritional data from USDA for each food item
            total_calories = 0
            descriptions = []
            
            for food_item in food_items:
                usda_data = await self._get_usda_nutrition(food_item)
                if usda_data:
                    total_calories += usda_data['calories']
                    descriptions.append(f"{food_item} ({usda_data['calories']} cal)")
                else:
                    # Fallback to OpenAI estimation if USDA data not available
                    estimated_cal = await self._estimate_food_calories(image_path, food_item)
                    total_calories += estimated_cal
                    descriptions.append(f"{food_item} (~{estimated_cal} cal)")
            
            description = f"Identified: {', '.join(descriptions)}"
            return round(total_calories, 1), description
            
        except Exception as e:
            print(f"Error in enhanced estimation: {e}")
            # Fallback to basic OpenAI estimation
            return await self._fallback_estimation(image_path)
    
    async def _identify_food_items(self, image_path: str) -> List[str]:
        """Use OpenAI to identify food items in the image"""
        try:
            base64_image = self.encode_image(image_path)
            
            prompt = """
            Analyze this food image and identify all food items visible. 
            List each food item on a separate line, starting with "FOOD:".
            
            Be specific about the food items (e.g., "grilled chicken breast", "white rice", "steamed broccoli").
            Only list foods that are clearly visible in the image.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.2
            )
            
            response_text = response.choices[0].message.content
            food_items = []
            
            for line in response_text.strip().split('\n'):
                if line.startswith("FOOD:"):
                    food_item = line.replace("FOOD:", "").strip()
                    if food_item:
                        food_items.append(food_item)
            
            return food_items
            
        except Exception as e:
            print(f"Error identifying food items: {e}")
            return []
    
    async def _get_usda_nutrition(self, food_item: str) -> Optional[Dict]:
        """Get nutritional data from USDA service"""
        try:
            # Search for the food item in USDA database
            foods = await usda_service.search_foods(food_item, page_size=5)
            
            if foods:
                # Return the first (most relevant) result
                return foods[0]
            
            return None
            
        except Exception as e:
            print(f"Error getting USDA nutrition for {food_item}: {e}")
            return None
    
    async def _estimate_food_calories(self, image_path: str, food_item: str) -> float:
        """Fallback estimation for individual food items"""
        try:
            base64_image = self.encode_image(image_path)
            
            prompt = f"""
            Estimate the calories for this specific food item: "{food_item}"
            
            Consider:
            - The portion size visible in the image
            - Cooking method (fried, grilled, raw, etc.)
            - Any visible ingredients or preparation
            
            Respond with just a number (e.g., "150").
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=50,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Extract number from response
            import re
            numbers = re.findall(r'\d+', response_text)
            if numbers:
                return float(numbers[0])
            
            return 200.0  # Default fallback
            
        except Exception as e:
            print(f"Error estimating calories for {food_item}: {e}")
            return 200.0
    
    async def _fallback_estimation(self, image_path: str) -> Tuple[float, str]:
        """Fallback to basic OpenAI estimation if enhanced method fails"""
        try:
            base64_image = self.encode_image(image_path)
            
            prompt = """
            Analyze this food image and provide a calorie estimation.
            
            Consider:
            - All food items visible
            - Portion sizes
            - Cooking methods
            - Hidden ingredients (oils, sauces, etc.)
            
            Respond in this exact format:
            CALORIES: [number]
            DESCRIPTION: [brief description]
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            lines = response_text.strip().split('\n')
            calories = None
            description = ""
            
            for line in lines:
                if line.startswith("CALORIES:"):
                    try:
                        calories = float(line.replace("CALORIES:", "").strip())
                    except ValueError:
                        calories = 300
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()
            
            if calories is None:
                calories = 300
            if not description:
                description = "Food items detected in the image"
            
            return round(calories, 1), description
            
        except Exception as e:
            print(f"Error in fallback estimation: {e}")
            return 300.0, "Unable to analyze image, using default estimate"
    
    def estimate_calories(self, image_path: str) -> Tuple[float, str]:
        """
        Synchronous wrapper for the async estimate_calories method
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (estimated_calories, food_description)
        """
        # Run the async method in a new event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use a different approach
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_estimate_calories(image_path))
                    return future.result()
            else:
                return loop.run_until_complete(self._async_estimate_calories(image_path))
        except RuntimeError:
            # No event loop running, create a new one
            return asyncio.run(self._async_estimate_calories(image_path))
    
    async def _async_estimate_calories(self, image_path: str) -> Tuple[float, str]:
        """Async version of estimate_calories"""
        return await self.estimate_calories(image_path)
