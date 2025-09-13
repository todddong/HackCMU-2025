"""
Comprehensive Nutrition API Client with Advanced Caching and Rate Limiting
This module provides a robust client for integrating with multiple nutrition APIs.

Features:
- Multiple API provider support (USDA, Edamam, Spoonacular)
- Intelligent caching with Redis/Memory
- Rate limiting and request throttling
- Data normalization across providers
- Error handling and retry logic
- Async/await support for high performance
"""

import asyncio
import aiohttp
import json
import time
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import redis
from functools import wraps
import backoff
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NutritionData:
    """Standardized nutrition data structure."""
    name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    sugar: float
    sodium: float
    serving_size: str
    serving_weight_g: float
    source: str
    confidence: float
    last_updated: datetime

@dataclass
class APIResponse:
    """Standardized API response structure."""
    success: bool
    data: List[NutritionData]
    total_results: int
    api_provider: str
    response_time_ms: float
    cache_hit: bool
    error_message: Optional[str] = None

class RateLimiter:
    """Advanced rate limiter with sliding window algorithm."""
    
    def __init__(self, max_requests: int, time_window: int):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            
            # Remove old requests outside the time window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    async def wait_for_slot(self) -> None:
        """Wait until a request slot becomes available."""
        while not await self.acquire():
            await asyncio.sleep(0.1)

class CacheManager:
    """Advanced caching system with multiple backends."""
    
    def __init__(self, cache_type: str = "memory", redis_url: str = None):
        """
        Initialize cache manager.
        
        Args:
            cache_type: Type of cache ('memory', 'redis', 'file')
            redis_url: Redis connection URL
        """
        self.cache_type = cache_type
        self.cache_ttl = 3600  # 1 hour default TTL
        
        if cache_type == "redis" and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Falling back to memory cache.")
                self.cache_type = "memory"
        
        if cache_type == "memory":
            self.memory_cache = {}
            self.cache_timestamps = {}
    
    def _generate_key(self, query: str, provider: str) -> str:
        """Generate cache key for query."""
        key_data = f"{provider}:{query.lower().strip()}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, query: str, provider: str) -> Optional[Dict]:
        """Get cached data."""
        key = self._generate_key(query, provider)
        
        if self.cache_type == "redis":
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    return json.loads(cached_data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        elif self.cache_type == "memory":
            if key in self.memory_cache:
                # Check if cache is still valid
                if time.time() - self.cache_timestamps[key] < self.cache_ttl:
                    return self.memory_cache[key]
                else:
                    # Remove expired cache
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
        
        return None
    
    async def set(self, query: str, provider: str, data: Dict) -> None:
        """Set cached data."""
        key = self._generate_key(query, provider)
        
        if self.cache_type == "redis":
            try:
                self.redis_client.setex(key, self.cache_ttl, json.dumps(data))
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        elif self.cache_type == "memory":
            self.memory_cache[key] = data
            self.cache_timestamps[key] = time.time()

class BaseNutritionAPI(ABC):
    """Abstract base class for nutrition API providers."""
    
    def __init__(self, api_key: str, rate_limiter: RateLimiter, cache_manager: CacheManager):
        self.api_key = api_key
        self.rate_limiter = rate_limiter
        self.cache_manager = cache_manager
        self.base_url = ""
        self.provider_name = ""
    
    @abstractmethod
    async def search_foods(self, query: str, limit: int = 20) -> List[NutritionData]:
        """Search for foods using the API."""
        pass
    
    @abstractmethod
    async def get_food_details(self, food_id: str) -> Optional[NutritionData]:
        """Get detailed information for a specific food."""
        pass
    
    def _normalize_nutrition_data(self, raw_data: Dict) -> NutritionData:
        """Normalize API-specific data to standard format."""
        # This would be implemented by each provider
        pass

class USDANutritionAPI(BaseNutritionAPI):
    """USDA FoodData Central API client."""
    
    def __init__(self, api_key: str, rate_limiter: RateLimiter, cache_manager: CacheManager):
        super().__init__(api_key, rate_limiter, cache_manager)
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.provider_name = "USDA"
    
    @backoff.on_exception(backoff.expo, aiohttp.ClientError, max_tries=3)
    async def search_foods(self, query: str, limit: int = 20) -> List[NutritionData]:
        """Search USDA FoodData Central."""
        # Check cache first
        cached_data = await self.cache_manager.get(query, self.provider_name)
        if cached_data:
            logger.info(f"Cache hit for USDA query: {query}")
            return [NutritionData(**item) for item in cached_data]
        
        await self.rate_limiter.wait_for_slot()
        
        url = f"{self.base_url}/foods/search"
        params = {
            "api_key": self.api_key,
            "query": query,
            "pageSize": limit,
            "dataType": ["Foundation", "SR Legacy", "Survey (FNDDS)", "Branded"]
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        nutrition_data = self._parse_usda_response(data)
                        
                        # Cache the results
                        await self.cache_manager.set(
                            query, 
                            self.provider_name, 
                            [asdict(item) for item in nutrition_data]
                        )
                        
                        return nutrition_data
                    else:
                        logger.error(f"USDA API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"USDA API request failed: {e}")
                return []
    
    async def get_food_details(self, food_id: str) -> Optional[NutritionData]:
        """Get detailed food information from USDA."""
        await self.rate_limiter.wait_for_slot()
        
        url = f"{self.base_url}/food/{food_id}"
        params = {"api_key": self.api_key}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_usda_food_details(data)
                    else:
                        logger.error(f"USDA food details error: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"USDA food details request failed: {e}")
                return None
    
    def _parse_usda_response(self, data: Dict) -> List[NutritionData]:
        """Parse USDA API response."""
        nutrition_data = []
        
        for food in data.get("foods", []):
            try:
                # Extract nutrition facts
                nutrients = {item["nutrient"]["name"]: item["amount"] 
                           for item in food.get("foodNutrients", [])}
                
                nutrition_data.append(NutritionData(
                    name=food.get("description", "Unknown"),
                    calories=nutrients.get("Energy", 0),
                    protein=nutrients.get("Protein", 0),
                    carbs=nutrients.get("Carbohydrate, by difference", 0),
                    fat=nutrients.get("Total lipid (fat)", 0),
                    fiber=nutrients.get("Fiber, total dietary", 0),
                    sugar=nutrients.get("Sugars, total including NLEA", 0),
                    sodium=nutrients.get("Sodium, Na", 0),
                    serving_size="100g",
                    serving_weight_g=100.0,
                    source=self.provider_name,
                    confidence=0.9,
                    last_updated=datetime.now()
                ))
            except Exception as e:
                logger.error(f"Error parsing USDA food data: {e}")
                continue
        
        return nutrition_data
    
    def _parse_usda_food_details(self, data: Dict) -> Optional[NutritionData]:
        """Parse detailed USDA food information."""
        try:
            nutrients = {item["nutrient"]["name"]: item["amount"] 
                        for item in data.get("foodNutrients", [])}
            
            return NutritionData(
                name=data.get("description", "Unknown"),
                calories=nutrients.get("Energy", 0),
                protein=nutrients.get("Protein", 0),
                carbs=nutrients.get("Carbohydrate, by difference", 0),
                fat=nutrients.get("Total lipid (fat)", 0),
                fiber=nutrients.get("Fiber, total dietary", 0),
                sugar=nutrients.get("Sugars, total including NLEA", 0),
                sodium=nutrients.get("Sodium, Na", 0),
                serving_size="100g",
                serving_weight_g=100.0,
                source=self.provider_name,
                confidence=0.95,
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing USDA food details: {e}")
            return None

class EdamamNutritionAPI(BaseNutritionAPI):
    """Edamam Nutrition API client."""
    
    def __init__(self, app_id: str, app_key: str, rate_limiter: RateLimiter, cache_manager: CacheManager):
        super().__init__(app_key, rate_limiter, cache_manager)
        self.app_id = app_id
        self.base_url = "https://api.edamam.com/api/nutrition-data"
        self.provider_name = "Edamam"
    
    async def search_foods(self, query: str, limit: int = 20) -> List[NutritionData]:
        """Edamam doesn't have a search API, so we return empty list."""
        return []
    
    async def get_food_details(self, food_id: str) -> Optional[NutritionData]:
        """Get nutrition data for a specific food from Edamam."""
        # This would require ingredient parsing
        return None

class SpoonacularNutritionAPI(BaseNutritionAPI):
    """Spoonacular API client."""
    
    def __init__(self, api_key: str, rate_limiter: RateLimiter, cache_manager: CacheManager):
        super().__init__(api_key, rate_limiter, cache_manager)
        self.base_url = "https://api.spoonacular.com"
        self.provider_name = "Spoonacular"
    
    async def search_foods(self, query: str, limit: int = 20) -> List[NutritionData]:
        """Search Spoonacular food database."""
        cached_data = await self.cache_manager.get(query, self.provider_name)
        if cached_data:
            logger.info(f"Cache hit for Spoonacular query: {query}")
            return [NutritionData(**item) for item in cached_data]
        
        await self.rate_limiter.wait_for_slot()
        
        url = f"{self.base_url}/food/products/search"
        params = {
            "apiKey": self.api_key,
            "query": query,
            "number": limit
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        nutrition_data = self._parse_spoonacular_response(data)
                        
                        await self.cache_manager.set(
                            query, 
                            self.provider_name, 
                            [asdict(item) for item in nutrition_data]
                        )
                        
                        return nutrition_data
                    else:
                        logger.error(f"Spoonacular API error: {response.status}")
                        return []
            except Exception as e:
                logger.error(f"Spoonacular API request failed: {e}")
                return []
    
    async def get_food_details(self, food_id: str) -> Optional[NutritionData]:
        """Get detailed food information from Spoonacular."""
        await self.rate_limiter.wait_for_slot()
        
        url = f"{self.base_url}/food/products/{food_id}"
        params = {"apiKey": self.api_key}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_spoonacular_food_details(data)
                    else:
                        logger.error(f"Spoonacular food details error: {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Spoonacular food details request failed: {e}")
                return None
    
    def _parse_spoonacular_response(self, data: Dict) -> List[NutritionData]:
        """Parse Spoonacular API response."""
        nutrition_data = []
        
        for product in data.get("products", []):
            try:
                nutrition = product.get("nutrition", {})
                nutrients = nutrition.get("nutrients", [])
                
                # Extract nutrition values
                nutrient_dict = {item["name"]: item["amount"] for item in nutrients}
                
                nutrition_data.append(NutritionData(
                    name=product.get("title", "Unknown"),
                    calories=nutrient_dict.get("Calories", 0),
                    protein=nutrient_dict.get("Protein", 0),
                    carbs=nutrient_dict.get("Carbohydrates", 0),
                    fat=nutrient_dict.get("Fat", 0),
                    fiber=nutrient_dict.get("Fiber", 0),
                    sugar=nutrient_dict.get("Sugar", 0),
                    sodium=nutrient_dict.get("Sodium", 0),
                    serving_size=nutrition.get("servingSize", "100g"),
                    serving_weight_g=100.0,
                    source=self.provider_name,
                    confidence=0.8,
                    last_updated=datetime.now()
                ))
            except Exception as e:
                logger.error(f"Error parsing Spoonacular product data: {e}")
                continue
        
        return nutrition_data
    
    def _parse_spoonacular_food_details(self, data: Dict) -> Optional[NutritionData]:
        """Parse detailed Spoonacular food information."""
        try:
            nutrition = data.get("nutrition", {})
            nutrients = nutrition.get("nutrients", [])
            nutrient_dict = {item["name"]: item["amount"] for item in nutrients}
            
            return NutritionData(
                name=data.get("title", "Unknown"),
                calories=nutrient_dict.get("Calories", 0),
                protein=nutrient_dict.get("Protein", 0),
                carbs=nutrient_dict.get("Carbohydrates", 0),
                fat=nutrient_dict.get("Fat", 0),
                fiber=nutrient_dict.get("Fiber", 0),
                sugar=nutrient_dict.get("Sugar", 0),
                sodium=nutrient_dict.get("Sodium", 0),
                serving_size=nutrition.get("servingSize", "100g"),
                serving_weight_g=100.0,
                source=self.provider_name,
                confidence=0.9,
                last_updated=datetime.now()
            )
        except Exception as e:
            logger.error(f"Error parsing Spoonacular food details: {e}")
            return None

class NutritionAPIClient:
    """
    Comprehensive nutrition API client that aggregates data from multiple providers.
    
    Features:
    - Multiple API provider support
    - Intelligent caching
    - Rate limiting
    - Data normalization
    - Error handling and fallbacks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the nutrition API client.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.config = config
        self.cache_manager = CacheManager(
            cache_type=config.get("cache_type", "memory"),
            redis_url=config.get("redis_url")
        )
        
        # Initialize rate limiters for each API
        self.rate_limiters = {
            "usda": RateLimiter(100, 3600),  # 100 requests per hour
            "edamam": RateLimiter(50, 3600),  # 50 requests per hour
            "spoonacular": RateLimiter(150, 3600)  # 150 requests per hour
        }
        
        # Initialize API clients
        self.apis = {}
        
        if config.get("usda_api_key"):
            self.apis["usda"] = USDANutritionAPI(
                config["usda_api_key"],
                self.rate_limiters["usda"],
                self.cache_manager
            )
        
        if config.get("edamam_app_id") and config.get("edamam_app_key"):
            self.apis["edamam"] = EdamamNutritionAPI(
                config["edamam_app_id"],
                config["edamam_app_key"],
                self.rate_limiters["edamam"],
                self.cache_manager
            )
        
        if config.get("spoonacular_api_key"):
            self.apis["spoonacular"] = SpoonacularNutritionAPI(
                config["spoonacular_api_key"],
                self.rate_limiters["spoonacular"],
                self.cache_manager
            )
        
        logger.info(f"Initialized nutrition API client with {len(self.apis)} providers")
    
    async def search_foods(self, query: str, limit: int = 20, providers: List[str] = None) -> APIResponse:
        """
        Search for foods across multiple providers.
        
        Args:
            query: Search query
            limit: Maximum number of results
            providers: List of providers to use (default: all available)
            
        Returns:
            Aggregated API response
        """
        if not providers:
            providers = list(self.apis.keys())
        
        start_time = time.time()
        all_results = []
        cache_hits = 0
        
        # Search across all providers concurrently
        tasks = []
        for provider in providers:
            if provider in self.apis:
                tasks.append(self._search_provider(provider, query, limit))
        
        if not tasks:
            return APIResponse(
                success=False,
                data=[],
                total_results=0,
                api_provider="none",
                response_time_ms=0,
                cache_hit=False,
                error_message="No API providers available"
            )
        
        # Execute all searches concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Provider {providers[i]} failed: {result}")
                continue
            
            if result.get("cache_hit"):
                cache_hits += 1
            
            all_results.extend(result.get("data", []))
        
        # Remove duplicates and sort by confidence
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.confidence, reverse=True)
        
        response_time = (time.time() - start_time) * 1000
        
        return APIResponse(
            success=len(sorted_results) > 0,
            data=sorted_results[:limit],
            total_results=len(sorted_results),
            api_provider=",".join(providers),
            response_time_ms=round(response_time, 2),
            cache_hit=cache_hits > 0
        )
    
    async def _search_provider(self, provider: str, query: str, limit: int) -> Dict:
        """Search a specific provider."""
        try:
            api = self.apis[provider]
            data = await api.search_foods(query, limit)
            
            return {
                "data": data,
                "cache_hit": False,  # This would be determined by the API
                "provider": provider
            }
        except Exception as e:
            logger.error(f"Error searching {provider}: {e}")
            return {"data": [], "cache_hit": False, "provider": provider}
    
    def _deduplicate_results(self, results: List[NutritionData]) -> List[NutritionData]:
        """Remove duplicate results based on name similarity."""
        unique_results = []
        seen_names = set()
        
        for result in results:
            # Simple deduplication based on name
            name_key = result.name.lower().strip()
            if name_key not in seen_names:
                seen_names.add(name_key)
                unique_results.append(result)
        
        return unique_results
    
    async def get_food_details(self, food_id: str, provider: str = "usda") -> Optional[NutritionData]:
        """Get detailed food information from a specific provider."""
        if provider not in self.apis:
            logger.error(f"Provider {provider} not available")
            return None
        
        try:
            api = self.apis[provider]
            return await api.get_food_details(food_id)
        except Exception as e:
            logger.error(f"Error getting food details from {provider}: {e}")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache_manager.cache_type == "memory":
            return {
                "cache_type": "memory",
                "cached_items": len(self.cache_manager.memory_cache),
                "cache_size_mb": sum(len(str(item)) for item in self.cache_manager.memory_cache.values()) / 1024 / 1024
            }
        elif self.cache_manager.cache_type == "redis":
            try:
                info = self.cache_manager.redis_client.info()
                return {
                    "cache_type": "redis",
                    "used_memory_mb": info.get("used_memory", 0) / 1024 / 1024,
                    "connected_clients": info.get("connected_clients", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0)
                }
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
                return {"cache_type": "redis", "error": str(e)}
        
        return {"cache_type": "unknown"}

# Example usage and configuration
if __name__ == "__main__":
    # Configuration example
    config = {
        "usda_api_key": "your_usda_api_key_here",
        "spoonacular_api_key": "your_spoonacular_api_key_here",
        "edamam_app_id": "your_edamam_app_id",
        "edamam_app_key": "your_edamam_app_key",
        "cache_type": "memory",  # or "redis"
        "redis_url": "redis://localhost:6379"
    }
    
    async def main():
        client = NutritionAPIClient(config)
        
        # Search for foods
        response = await client.search_foods("apple", limit=10)
        print(f"Found {response.total_results} results")
        print(f"Response time: {response.response_time_ms}ms")
        print(f"Cache hit: {response.cache_hit}")
        
        # Get cache statistics
        stats = client.get_cache_stats()
        print(f"Cache stats: {stats}")
    
    # Run the example
    # asyncio.run(main())

