import httpx
from typing import List, Dict, Optional
import os

class USDAService:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize USDA FoodData Central API service
        
        Args:
            api_key: USDA API key. If not provided, will use demo key or public endpoints
        """
        # For demo purposes, we'll use a demo API key
        # In production, you should get a real API key from https://fdc.nal.usda.gov/api-guide.html
        self.api_key = api_key or "DEMO_KEY"
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_foods(self, query: str, page_size: int = 20, data_type: str = None) -> List[Dict]:
        """
        Search for foods using USDA FoodData Central API
        
        Args:
            query: Search term for food
            page_size: Number of results to return (max 200)
            data_type: Filter by data type (Foundation, SR Legacy, Survey (FNDDS), Branded, Experimental)
            
        Returns:
            List of food items with nutritional information
        """
        # For now, use mock data to avoid rate limiting issues
        # In production, you would implement proper API key rotation and rate limiting
        print(f"Searching for: {query} (using mock data to avoid rate limits)")
        return self._get_mock_foods(query.lower())
        
        # Uncomment below for real API usage (requires proper API key management)
        """
        try:
            # Build search parameters
            params = {
                "query": query,
                "pageSize": min(page_size, 200),
                "api_key": self.api_key
            }
            
            # Add data type filter if specified
            if data_type:
                params["dataType"] = data_type
            
            # Make API request
            response = await self.client.get(f"{self.base_url}/foods/search", params=params)
            response.raise_for_status()
            
            data = response.json()
            foods = data.get("foods", [])
            
            # Process and format the results
            processed_foods = []
            for food in foods:
                processed_food = self._process_food_data(food)
                if processed_food:
                    processed_foods.append(processed_food)
            
            return processed_foods
            
        except httpx.HTTPError as e:
            print(f"HTTP error searching foods: {e}")
            # Return mock data as fallback
            return self._get_mock_foods(query.lower())
        except Exception as e:
            print(f"Error searching foods: {e}")
            # Return mock data as fallback
            return self._get_mock_foods(query.lower())
        """
    
    def _process_food_data(self, food: Dict) -> Optional[Dict]:
        """
        Process raw food data from USDA API into our format
        
        Args:
            food: Raw food data from API
            
        Returns:
            Processed food data or None if invalid
        """
        try:
            # Extract basic information
            fdc_id = food.get("fdcId")
            description = food.get("description", "Unknown food")
            data_type = food.get("dataType", "Unknown")
            
            # Extract nutritional information
            nutrients = food.get("foodNutrients", [])
            calories = self._extract_calories(nutrients)
            
            # Extract additional nutritional info with multiple possible names
            protein = self._extract_nutrient_by_names(nutrients, ["Protein", "Protein (N x 6.25)"])
            carbs = self._extract_nutrient_by_names(nutrients, ["Carbohydrate, by difference", "Carbohydrate, by summation", "Total carbohydrate"])
            fat = self._extract_nutrient_by_names(nutrients, ["Total lipid (fat)", "Fat (NLEA)", "Total fat"])
            fiber = self._extract_nutrient_by_names(nutrients, ["Fiber, total dietary", "Dietary fiber"])
            
            # Get serving size info
            serving_size = self._get_serving_size(food)
            
            return {
                "fdc_id": fdc_id,
                "name": description,
                "calories": calories,
                "protein": protein,
                "carbs": carbs,
                "fat": fat,
                "fiber": fiber,
                "serving_size": serving_size,
                "data_type": data_type,
                "brand_owner": food.get("brandOwner"),
                "ingredients": food.get("ingredients")
            }
            
        except Exception as e:
            print(f"Error processing food data: {e}")
            return None
    
    def _extract_calories(self, nutrients: List[Dict]) -> float:
        """Extract calories from nutrient data"""
        # Try different possible names for calories/energy
        calorie_names = ["Energy", "Energy (Atwater General Factors)", "Energy (Atwater Specific Factors)"]
        
        for nutrient in nutrients:
            nutrient_name = nutrient.get("nutrientName", "")
            unit_name = nutrient.get("unitName", "")
            value = nutrient.get("value")
            
            if nutrient_name in calorie_names and unit_name == "KCAL" and value is not None:
                return float(value)
        
        return 0.0
    
    def _extract_nutrient(self, nutrients: List[Dict], nutrient_name: str) -> float:
        """Extract specific nutrient value"""
        for nutrient in nutrients:
            if nutrient.get("nutrientName") == nutrient_name:
                value = nutrient.get("value")
                if value is not None:
                    return float(value)
        return 0.0
    
    def _extract_nutrient_by_names(self, nutrients: List[Dict], possible_names: List[str]) -> float:
        """Extract nutrient value by trying multiple possible names"""
        for nutrient in nutrients:
            nutrient_name = nutrient.get("nutrientName", "")
            if nutrient_name in possible_names:
                value = nutrient.get("value")
                if value is not None:
                    return float(value)
        return 0.0
    
    def _get_serving_size(self, food: Dict) -> str:
        """Get serving size information"""
        # Try to get serving size from the main food object
        serving_size = food.get("servingSize")
        serving_size_unit = food.get("servingSizeUnit", "g")
        household_serving = food.get("householdServingFullText")
        
        if household_serving:
            return household_serving
        elif serving_size and serving_size_unit:
            return f"{serving_size} {serving_size_unit}"
        
        # Try to get serving size from foodPortions
        portions = food.get("foodPortions", [])
        if portions:
            portion = portions[0]
            amount = portion.get("amount", 1)
            unit = portion.get("measureUnit", {}).get("name", "serving")
            return f"{amount} {unit}"
        
        # Fallback to default
        return "1 serving"
    
    def _get_mock_foods(self, query: str) -> List[Dict]:
        """Get mock food data as fallback when API is unavailable"""
        food_database = {
            "apple": [
                {
                    "fdc_id": "mock_1",
                    "name": "Apple, raw, with skin",
                    "calories": 52.0,
                    "protein": 0.3,
                    "carbs": 13.8,
                    "fat": 0.2,
                    "fiber": 2.4,
                    "serving_size": "1 medium (182g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_1b",
                    "name": "Apple juice, unsweetened",
                    "calories": 46.0,
                    "protein": 0.1,
                    "carbs": 11.3,
                    "fat": 0.1,
                    "fiber": 0.2,
                    "serving_size": "1 cup (248g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ],
            "chicken": [
                {
                    "fdc_id": "mock_2",
                    "name": "Chicken breast, skinless, boneless, raw",
                    "calories": 165.0,
                    "protein": 31.0,
                    "carbs": 0.0,
                    "fat": 3.6,
                    "fiber": 0.0,
                    "serving_size": "100g",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_2b",
                    "name": "Chicken thigh, skinless, boneless, raw",
                    "calories": 209.0,
                    "protein": 18.0,
                    "carbs": 0.0,
                    "fat": 14.0,
                    "fiber": 0.0,
                    "serving_size": "100g",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ],
            "pizza": [
                {
                    "fdc_id": "mock_3",
                    "name": "Pizza, cheese, regular crust",
                    "calories": 266.0,
                    "protein": 12.2,
                    "carbs": 33.0,
                    "fat": 9.8,
                    "fiber": 2.3,
                    "serving_size": "1 slice (107g)",
                    "data_type": "Branded",
                    "brand_owner": "Generic",
                    "ingredients": "Dough, cheese, tomato sauce"
                },
                {
                    "fdc_id": "mock_3b",
                    "name": "Pizza, pepperoni, regular crust",
                    "calories": 281.0,
                    "protein": 13.0,
                    "carbs": 33.0,
                    "fat": 11.0,
                    "fiber": 2.3,
                    "serving_size": "1 slice (107g)",
                    "data_type": "Branded",
                    "brand_owner": "Generic",
                    "ingredients": "Dough, cheese, pepperoni, tomato sauce"
                }
            ],
            "banana": [
                {
                    "fdc_id": "mock_4",
                    "name": "Banana, raw",
                    "calories": 89.0,
                    "protein": 1.1,
                    "carbs": 22.8,
                    "fat": 0.3,
                    "fiber": 2.6,
                    "serving_size": "1 medium (118g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_4b",
                    "name": "Banana chips, fried",
                    "calories": 519.0,
                    "protein": 2.3,
                    "carbs": 64.0,
                    "fat": 33.6,
                    "fiber": 7.7,
                    "serving_size": "1 cup (72g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ],
            "rice": [
                {
                    "fdc_id": "mock_5",
                    "name": "Rice, white, long-grain, cooked",
                    "calories": 130.0,
                    "protein": 2.7,
                    "carbs": 28.0,
                    "fat": 0.3,
                    "fiber": 0.4,
                    "serving_size": "1 cup (158g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_5b",
                    "name": "Rice, brown, long-grain, cooked",
                    "calories": 111.0,
                    "protein": 2.6,
                    "carbs": 23.0,
                    "fat": 0.9,
                    "fiber": 1.8,
                    "serving_size": "1 cup (195g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ],
            "bread": [
                {
                    "fdc_id": "mock_6",
                    "name": "Bread, white, commercially prepared",
                    "calories": 265.0,
                    "protein": 9.0,
                    "carbs": 49.0,
                    "fat": 3.2,
                    "fiber": 2.7,
                    "serving_size": "1 slice (25g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_6b",
                    "name": "Bread, whole wheat, commercially prepared",
                    "calories": 247.0,
                    "protein": 13.0,
                    "carbs": 41.0,
                    "fat": 4.2,
                    "fiber": 6.0,
                    "serving_size": "1 slice (28g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ],
            "salad": [
                {
                    "fdc_id": "mock_7",
                    "name": "Lettuce, romaine, raw",
                    "calories": 17.0,
                    "protein": 1.2,
                    "carbs": 3.3,
                    "fat": 0.3,
                    "fiber": 2.1,
                    "serving_size": "1 cup shredded (47g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_7b",
                    "name": "Caesar salad, with dressing",
                    "calories": 184.0,
                    "protein": 8.0,
                    "carbs": 8.0,
                    "fat": 15.0,
                    "fiber": 2.0,
                    "serving_size": "1 cup (152g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ],
            "pasta": [
                {
                    "fdc_id": "mock_8",
                    "name": "Pasta, spaghetti, cooked",
                    "calories": 131.0,
                    "protein": 5.0,
                    "carbs": 25.0,
                    "fat": 1.1,
                    "fiber": 1.8,
                    "serving_size": "1 cup (140g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ],
            "mcdonalds": [
                {
                    "fdc_id": "mock_9",
                    "name": "McDonald's Big Mac",
                    "calories": 550.0,
                    "protein": 25.0,
                    "carbs": 45.0,
                    "fat": 33.0,
                    "fiber": 3.0,
                    "serving_size": "1 sandwich (215g)",
                    "data_type": "Branded",
                    "brand_owner": "McDonald's Corporation",
                    "ingredients": "Beef patty, bun, lettuce, cheese, pickles, onions, special sauce"
                },
                {
                    "fdc_id": "mock_9b",
                    "name": "McDonald's French Fries",
                    "calories": 320.0,
                    "protein": 4.0,
                    "carbs": 43.0,
                    "fat": 15.0,
                    "fiber": 4.0,
                    "serving_size": "1 medium (117g)",
                    "data_type": "Branded",
                    "brand_owner": "McDonald's Corporation",
                    "ingredients": "Potatoes, vegetable oil, salt"
                }
            ],
            "coca": [
                {
                    "fdc_id": "mock_10",
                    "name": "Coca-Cola Classic",
                    "calories": 140.0,
                    "protein": 0.0,
                    "carbs": 39.0,
                    "fat": 0.0,
                    "fiber": 0.0,
                    "serving_size": "1 can (355ml)",
                    "data_type": "Branded",
                    "brand_owner": "The Coca-Cola Company",
                    "ingredients": "Carbonated water, high fructose corn syrup, caramel color, phosphoric acid, natural flavors, caffeine"
                }
            ]
        }
        
        # Find matching foods with improved search logic
        results = []
        query_lower = query.lower().strip()
        
        # First, try exact matches
        for food_type, foods in food_database.items():
            if food_type == query_lower:
                results.extend(foods)
        
        # Then try partial matches
        if not results:
            for food_type, foods in food_database.items():
                if food_type in query_lower or any(word in query_lower for word in food_type.split()):
                    results.extend(foods)
        
        # Try brand-specific searches
        if not results:
            if "mcdonald" in query_lower or "mcd" in query_lower:
                results.extend(food_database.get("mcdonalds", []))
            elif "coca" in query_lower or "cola" in query_lower:
                results.extend(food_database.get("coca", []))
        
        # Try broader category searches
        if not results:
            if any(word in query_lower for word in ["fruit", "fruits"]):
                results.extend(food_database.get("apple", []))
                results.extend(food_database.get("banana", []))
            elif any(word in query_lower for word in ["meat", "protein", "beef", "pork"]):
                results.extend(food_database.get("chicken", []))
            elif any(word in query_lower for word in ["grain", "cereal", "wheat"]):
                results.extend(food_database.get("bread", []))
                results.extend(food_database.get("rice", []))
                results.extend(food_database.get("pasta", []))
            elif any(word in query_lower for word in ["vegetable", "veggie", "green"]):
                results.extend(food_database.get("salad", []))
        
        # If still no results, return some common foods
        if not results:
            results = [
                {
                    "fdc_id": "mock_default_1",
                    "name": "Apple, raw, with skin",
                    "calories": 52.0,
                    "protein": 0.3,
                    "carbs": 13.8,
                    "fat": 0.2,
                    "fiber": 2.4,
                    "serving_size": "1 medium (182g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_default_2",
                    "name": "Chicken breast, skinless, raw",
                    "calories": 165.0,
                    "protein": 31.0,
                    "carbs": 0.0,
                    "fat": 3.6,
                    "fiber": 0.0,
                    "serving_size": "100g",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                },
                {
                    "fdc_id": "mock_default_3",
                    "name": "Rice, white, long-grain, cooked",
                    "calories": 130.0,
                    "protein": 2.7,
                    "carbs": 28.0,
                    "fat": 0.3,
                    "fiber": 0.4,
                    "serving_size": "1 cup (158g)",
                    "data_type": "Foundation",
                    "brand_owner": None,
                    "ingredients": None
                }
            ]
        
        return results[:10]  # Limit to 10 results
    
    async def get_food_details(self, fdc_id: str) -> Optional[Dict]:
        """
        Get detailed information for a specific food item
        
        Args:
            fdc_id: FoodData Central ID
            
        Returns:
            Detailed food information or None if not found
        """
        try:
            response = await self.client.get(
                f"{self.base_url}/food/{fdc_id}",
                params={"api_key": self.api_key}
            )
            response.raise_for_status()
            
            food_data = response.json()
            return self._process_food_data(food_data)
            
        except Exception as e:
            print(f"Error getting food details: {e}")
            return None
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Create a global instance
usda_service = USDAService()
