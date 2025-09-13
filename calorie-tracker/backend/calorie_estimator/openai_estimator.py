import base64
import os
from typing import Tuple
from openai import OpenAI
from PIL import Image
import io
from .base import CalorieEstimator

class OpenAICalorieEstimator(CalorieEstimator):
    """OpenAI GPT-4 Vision API calorie estimator for real food photo analysis"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the OpenAI calorie estimator
        
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
        Converts image to JPEG format to ensure compatibility
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Open and convert image to RGB JPEG format
            with Image.open(image_path) as img:
                # Convert to RGB if necessary (handles RGBA, P, etc.)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes buffer as JPEG
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=85, optimize=True)
                buffer.seek(0)
                
                # Encode to base64
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
                
        except Exception as e:
            # Fallback to original method if PIL fails
            print(f"Warning: PIL conversion failed ({e}), using original image")
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    
    def estimate_calories(self, image_path: str) -> Tuple[float, str]:
        """
        Estimate calories from food image using OpenAI GPT-4 Vision
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (estimated_calories, food_description)
        """
        try:
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Create the prompt for calorie estimation
            prompt = """
            Analyze this food image and provide a detailed calorie estimation. Please:
            
            1. Identify all food items visible in the image
            2. Estimate the portion sizes
            3. Calculate the approximate total calories
            4. Provide a brief description of what you see
            
            Consider:
            - Different types of food (proteins, carbs, fats, vegetables, etc.)
            - Cooking methods (fried, grilled, raw, etc.)
            - Portion sizes relative to common serving sizes
            - Hidden ingredients (oils, sauces, etc.)
            
            Respond in this exact format:
            CALORIES: [number]
            DESCRIPTION: [brief description of the food items and portions]
            """
            
            # Make the API call using GPT-4o-mini for cost efficiency
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=500,
                    temperature=0.3
                )
            except Exception as e:
                error_msg = str(e)
                if "rate_limit_exceeded" in error_msg or "429" in error_msg:
                    return 0.0, "Rate limit exceeded. Please wait a moment and try again. Consider upgrading your OpenAI plan for higher limits."
                elif "invalid_image_format" in error_msg or "400" in error_msg:
                    return 0.0, "Image format not supported. Please use PNG, JPEG, GIF, or WebP format."
                else:
                    return 0.0, f"Error analyzing image: {error_msg}"
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Extract calories and description
            lines = response_text.strip().split('\n')
            calories = None
            description = ""
            
            for line in lines:
                if line.startswith("CALORIES:"):
                    try:
                        calories = float(line.replace("CALORIES:", "").strip())
                    except ValueError:
                        calories = 300  # Default fallback
                elif line.startswith("DESCRIPTION:"):
                    description = line.replace("DESCRIPTION:", "").strip()
            
            # Fallback if parsing fails
            if calories is None:
                calories = 300
            if not description:
                description = "Food items detected in the image"
            
            return round(calories, 1), description
            
        except Exception as e:
            # Fallback to a reasonable default if API call fails
            print(f"Error calling OpenAI API: {e}")
            return 300.0, "Unable to analyze image, using default estimate"
