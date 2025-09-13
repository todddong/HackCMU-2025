#!/usr/bin/env python3
"""
Configuration file for the nutriAI application
This file contains all configuration settings including API keys and feature flags.
"""

import os
from typing import Optional

class Config:
    """Application configuration settings"""
    
    # OpenAI API Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "sk-dummy-key-replace-with-real-key-later")
    
    # USDA API Configuration  
    USDA_API_KEY: str = os.getenv("USDA_API_KEY", "DEMO_KEY")
    
    # Feature Flags
    ENABLE_GPT_SEARCH: bool = True
    ENABLE_GPT_CALORIE_ESTIMATION: bool = True
    ENABLE_USDA_API: bool = False  # Set to True when you have a real USDA API key
    
    # API Endpoints
    USDA_BASE_URL: str = "https://api.nal.usda.gov/fdc/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # Search Configuration
    DEFAULT_SEARCH_PAGE_SIZE: int = 20
    MAX_SEARCH_RESULTS: int = 50
    
    # Development Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    MOCK_MODE: bool = OPENAI_API_KEY == "sk-dummy-key-replace-with-real-key-later"
    
    @classmethod
    def is_gpt_available(cls) -> bool:
        """Check if GPT features are available (real API key configured)"""
        return cls.OPENAI_API_KEY != "sk-dummy-key-replace-with-real-key-later" and cls.OPENAI_API_KEY is not None
    
    @classmethod
    def get_openai_api_key(cls) -> Optional[str]:
        """Get OpenAI API key, returning None if using dummy key"""
        if cls.OPENAI_API_KEY == "sk-dummy-key-replace-with-real-key-later":
            return None
        return cls.OPENAI_API_KEY

# Global configuration instance
config = Config()
