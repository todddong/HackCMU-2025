#!/usr/bin/env python3
"""
GPT-Powered Food Search Wrapper
This module provides intelligent food search capabilities using OpenAI's GPT models
combined with USDA nutritional data for enhanced search results.
"""

import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from openai import OpenAI
import httpx
from config import config

@dataclass
class SearchResult:
    """Structured search result with enhanced information"""
    fdc_id: str
    name: str
    description: str
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    serving_size: str
    data_type: str
    brand_owner: Optional[str]
    ingredients: Optional[str]
    confidence_score: float
    search_reasoning: str
    nutritional_highlights: List[str]

class GPTSearchWrapper:
    """
    GPT-powered food search wrapper that enhances USDA search results
    with AI-powered food identification, nutritional analysis, and smart recommendations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT search wrapper
        
        Args:
            api_key: OpenAI API key. If not provided, uses configuration
        """
        # Use configuration for API keys
        self.api_key = api_key or config.get_openai_api_key()
        self.client = OpenAI(api_key=self.api_key) if config.is_gpt_available() else None
        self.usda_base_url = config.USDA_BASE_URL
        self.usda_api_key = config.USDA_API_KEY
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Enhanced food knowledge base for better search results
        self.food_knowledge = {
            "common_foods": {
                "apple": {"category": "fruit", "keywords": ["fruit", "healthy", "fiber", "vitamin c"]},
                "banana": {"category": "fruit", "keywords": ["fruit", "potassium", "energy", "portable"]},
                "chicken": {"category": "protein", "keywords": ["meat", "protein", "lean", "healthy"]},
                "rice": {"category": "grain", "keywords": ["carbohydrate", "staple", "energy", "gluten-free"]},
                "pizza": {"category": "processed", "keywords": ["comfort", "cheese", "carbs", "indulgent"]},
                "salad": {"category": "vegetable", "keywords": ["healthy", "low-calorie", "vitamins", "fresh"]},
                "bread": {"category": "grain", "keywords": ["carbohydrate", "staple", "fiber", "energy"]},
                "pasta": {"category": "grain", "keywords": ["carbohydrate", "comfort", "italian", "energy"]}
            },
            "brands": {
                "mcdonalds": {"category": "fast_food", "keywords": ["burger", "fries", "fast", "convenient"]},
                "coca-cola": {"category": "beverage", "keywords": ["soda", "sweet", "caffeine", "carbonated"]},
                "lays": {"category": "snack", "keywords": ["chips", "salty", "crunchy", "snack"]},
                "starbucks": {"category": "beverage", "keywords": ["coffee", "caffeine", "premium", "drink"]}
            },
            "nutritional_categories": {
                "high_protein": ["chicken", "beef", "fish", "eggs", "tofu", "beans"],
                "low_calorie": ["lettuce", "cucumber", "celery", "broccoli", "spinach"],
                "high_fiber": ["beans", "lentils", "berries", "avocado", "quinoa"],
                "healthy_fats": ["avocado", "nuts", "olive oil", "salmon", "seeds"]
            }
        }
    
    async def search_foods(self, query: str, page_size: int = 20, data_type: str = None) -> List[Dict]:
        """
        Enhanced food search using GPT for intelligent query processing and result enhancement
        
        Args:
            query: Search query from user
            page_size: Number of results to return
            data_type: Filter by data type
            
        Returns:
            List of enhanced food search results
        """
        try:
            # Step 1: Process query with GPT for better understanding
            processed_query = await self._process_query_with_gpt(query)
            
            # Step 2: Get USDA search results
            usda_results = await self._get_usda_results(processed_query, page_size, data_type)
            
            # Step 3: Enhance results with GPT analysis
            enhanced_results = await self._enhance_results_with_gpt(query, usda_results)
            
            return enhanced_results
            
        except Exception as e:
            print(f"Error in GPT search: {e}")
            # Fallback to mock data
            return self._get_fallback_results(query)
    
    async def _process_query_with_gpt(self, query: str) -> str:
        """
        Use GPT to process and enhance the search query for better USDA API results
        """
        if not self.client:
            # Use dummy processing when no real API key
            return self._dummy_query_processing(query)
        
        try:
            prompt = f"""
            Analyze this food search query and provide an optimized search term for USDA FoodData Central API.
            
            Original query: "{query}"
            
            Consider:
            1. Common food names and synonyms
            2. Brand names vs generic terms
            3. Cooking methods (raw, cooked, fried, etc.)
            4. Food categories and types
            
            Return only the optimized search term, nothing else.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"GPT query processing failed: {e}")
            return query
    
    def _dummy_query_processing(self, query: str) -> str:
        """
        Dummy query processing for development without real API key
        """
        query_lower = query.lower()
        
        # Simple keyword mapping for common searches
        mappings = {
            "mcd": "mcdonalds",
            "mcdonald": "mcdonalds", 
            "coca": "coca-cola",
            "cola": "coca-cola",
            "lays": "lays chips",
            "chips": "potato chips",
            "burger": "hamburger",
            "fries": "french fries",
            "pizza": "pizza cheese",
            "salad": "lettuce salad",
            "chicken": "chicken breast",
            "rice": "white rice",
            "bread": "white bread",
            "pasta": "spaghetti pasta"
        }
        
        for key, value in mappings.items():
            if key in query_lower:
                return value
        
        return query
    
    async def _get_usda_results(self, query: str, page_size: int, data_type: str) -> List[Dict]:
        """
        Get results from USDA API (currently using mock data)
        """
        # For now, use mock data to avoid rate limiting
        # In production, implement proper USDA API calls
        print(f"🔍 GPT-enhanced search for: {query}")
        return self._get_mock_usda_results(query.lower())
    
    async def _enhance_results_with_gpt(self, original_query: str, usda_results: List[Dict]) -> List[Dict]:
        """
        Use GPT to enhance USDA results with better descriptions and nutritional insights
        """
        if not self.client:
            # Use dummy enhancement when no real API key
            return self._dummy_result_enhancement(original_query, usda_results)
        
        try:
            # Prepare results for GPT analysis
            results_summary = []
            for result in usda_results[:5]:  # Analyze top 5 results
                results_summary.append({
                    "name": result.get("name", ""),
                    "calories": result.get("calories", 0),
                    "protein": result.get("protein", 0),
                    "carbs": result.get("carbs", 0),
                    "fat": result.get("fat", 0)
                })
            
            prompt = f"""
            Analyze these food search results for the query "{original_query}" and provide enhanced descriptions, nutritional insights, and macro predictions.
            
            Results: {json.dumps(results_summary, indent=2)}
            
            For each result, provide:
            1. A more descriptive name
            2. Nutritional highlights (2-3 key points)
            3. Confidence score (0-1) for how well it matches the query
            4. Brief reasoning for why this result is relevant
            5. Predicted macros (protein, carbs, fat) based on the food type and calories
            
            Return as JSON array with fields: name, description, nutritional_highlights, confidence_score, reasoning, predicted_protein, predicted_carbs, predicted_fat
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse GPT response and merge with USDA data
            gpt_analysis = json.loads(response.choices[0].message.content)
            
            enhanced_results = []
            for i, result in enumerate(usda_results):
                if i < len(gpt_analysis):
                    analysis = gpt_analysis[i]
                    enhanced_result = result.copy()
                    
                    # Use predicted macros if available, otherwise keep original values
                    predicted_protein = analysis.get("predicted_protein")
                    predicted_carbs = analysis.get("predicted_carbs")
                    predicted_fat = analysis.get("predicted_fat")
                    
                    enhanced_result.update({
                        "description": analysis.get("description", result.get("name", "")),
                        "nutritional_highlights": analysis.get("nutritional_highlights", []),
                        "confidence_score": analysis.get("confidence_score", 0.8),
                        "search_reasoning": analysis.get("reasoning", "Relevant food match"),
                        "protein": predicted_protein if predicted_protein is not None else result.get("protein", 0),
                        "carbs": predicted_carbs if predicted_carbs is not None else result.get("carbs", 0),
                        "fat": predicted_fat if predicted_fat is not None else result.get("fat", 0)
                    })
                    enhanced_results.append(enhanced_result)
                else:
                    enhanced_results.append(result)
            
            return enhanced_results
            
        except Exception as e:
            print(f"GPT result enhancement failed: {e}")
            return usda_results
    
    def _dummy_result_enhancement(self, original_query: str, usda_results: List[Dict]) -> List[Dict]:
        """
        Dummy result enhancement for development without real API key
        """
        enhanced_results = []
        
        for result in usda_results:
            enhanced_result = result.copy()
            
            # Add dummy enhancements based on food type
            name = result.get("name", "").lower()
            calories = result.get("calories", 0)
            protein = result.get("protein", 0)
            carbs = result.get("carbs", 0)
            fat = result.get("fat", 0)
            
            # Generate nutritional highlights
            highlights = []
            if protein > 20:
                highlights.append("High protein content")
            if calories < 100:
                highlights.append("Low calorie option")
            elif calories > 400:
                highlights.append("High energy food")
            if carbs > 30:
                highlights.append("Good source of carbohydrates")
            if fat < 5:
                highlights.append("Low fat content")
            
            # Enhanced macro predictions based on food type
            enhanced_protein = protein
            enhanced_carbs = carbs
            enhanced_fat = fat
            
            # Apply AI-like macro predictions based on food characteristics
            if "chicken" in name.lower() or "beef" in name.lower() or "fish" in name.lower():
                enhanced_protein = max(protein, calories * 0.25 / 4)  # 25% protein
                enhanced_fat = max(fat, calories * 0.15 / 9)  # 15% fat
            elif "rice" in name.lower() or "pasta" in name.lower() or "bread" in name.lower():
                enhanced_carbs = max(carbs, calories * 0.70 / 4)  # 70% carbs
                enhanced_protein = max(protein, calories * 0.10 / 4)  # 10% protein
            elif "pizza" in name.lower() or "burger" in name.lower():
                enhanced_fat = max(fat, calories * 0.35 / 9)  # 35% fat
                enhanced_carbs = max(carbs, calories * 0.40 / 4)  # 40% carbs
                enhanced_protein = max(protein, calories * 0.20 / 4)  # 20% protein
            
            # Generate confidence score based on query match
            confidence = 0.8
            query_words = original_query.lower().split()
            for word in query_words:
                if word in name:
                    confidence += 0.1
            
            confidence = min(confidence, 1.0)
            
            enhanced_result.update({
                "description": f"🤖 AI-Enhanced: {result.get('name', '')}",
                "nutritional_highlights": highlights,
                "confidence_score": round(confidence, 2),
                "search_reasoning": f"Matches search criteria for '{original_query}'",
                "protein": round(enhanced_protein, 1),
                "carbs": round(enhanced_carbs, 1),
                "fat": round(enhanced_fat, 1)
            })
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    def _get_mock_usda_results(self, query: str) -> List[Dict]:
        """
        Get mock USDA results for development
        """
        # Enhanced mock database with more foods
        food_database = {
            "apple": [
                {
                    "fdc_id": "gpt_1",
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
                }
            ],
            "chicken": [
                {
                    "fdc_id": "gpt_2",
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
                }
            ],
            "pizza": [
                {
                    "fdc_id": "gpt_3",
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
                }
            ],
            "mcdonalds": [
                {
                    "fdc_id": "gpt_4",
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
                }
            ],
            "lays": [
                {
                    "fdc_id": "gpt_5",
                    "name": "Lay's Classic Potato Chips",
                    "calories": 160.0,
                    "protein": 2.0,
                    "carbs": 15.0,
                    "fat": 10.0,
                    "fiber": 1.0,
                    "serving_size": "1 oz (28g)",
                    "data_type": "Branded",
                    "brand_owner": "Frito-Lay",
                    "ingredients": "Potatoes, vegetable oil, salt"
                }
            ]
        }
        
        # Find matching foods
        results = []
        for food_type, foods in food_database.items():
            if food_type in query or any(word in query for word in food_type.split()):
                results.extend(foods)
        
        # If no specific matches, return some common foods
        if not results:
            results = [
                {
                    "fdc_id": "gpt_default_1",
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
                }
            ]
        
        return results[:10]
    
    def _get_fallback_results(self, query: str) -> List[Dict]:
        """
        Get fallback results when GPT search fails
        """
        return [
            {
                "fdc_id": "fallback_1",
                "name": f"Search result for '{query}'",
                "description": f"🤖 AI search result for '{query}'",
                "calories": 100.0,
                "protein": 5.0,
                "carbs": 15.0,
                "fat": 3.0,
                "fiber": 2.0,
                "serving_size": "1 serving",
                "data_type": "Foundation",
                "brand_owner": None,
                "ingredients": None,
                "nutritional_highlights": ["AI-powered search result"],
                "confidence_score": 0.5,
                "search_reasoning": "Fallback result due to search limitations"
            }
        ]
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()

# Create global instance with dummy API key
gpt_search_wrapper = GPTSearchWrapper()
