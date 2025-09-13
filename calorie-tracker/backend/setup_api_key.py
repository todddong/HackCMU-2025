#!/usr/bin/env python3
"""
Setup script for OpenAI API key configuration.
Run this script to set up your OpenAI API key for the calorie tracker.
"""

import os
import sys

def setup_api_key():
    """Interactive setup for OpenAI API key"""
    print("🔑 OpenAI API Key Setup for Calorie Tracker")
    print("=" * 50)
    
    # Check if API key is already set
    current_key = os.getenv("OPENAI_API_KEY")
    if current_key:
        print(f"✅ API key is already set: {current_key[:8]}...")
        choice = input("Do you want to update it? (y/n): ").lower().strip()
        if choice != 'y':
            print("Keeping existing API key.")
            return
    
    print("\n📋 Instructions:")
    print("1. Go to https://platform.openai.com/api-keys")
    print("2. Create a new API key")
    print("3. Copy the key (it starts with 'sk-')")
    print("4. Paste it below")
    print("\n💡 Note: GPT-4o-mini is very cost-effective (~$0.00015 per image)")
    
    # Get API key from user
    api_key = input("\n🔑 Enter your OpenAI API key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    if not api_key.startswith('sk-'):
        print("⚠️  Warning: API key doesn't start with 'sk-'. Are you sure it's correct?")
        confirm = input("Continue anyway? (y/n): ").lower().strip()
        if confirm != 'y':
            print("❌ Setup cancelled.")
            return
    
    # Set environment variable for current session
    os.environ["OPENAI_API_KEY"] = api_key
    
    # Create .env file for persistence
    env_file = ".env"
    with open(env_file, "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
    
    print(f"\n✅ API key set successfully!")
    print(f"📁 Saved to {env_file} for future sessions")
    print(f"🔑 Key: {api_key[:8]}...{api_key[-4:]}")
    
    # Test the setup
    print("\n🧪 Testing API connection...")
    try:
        from calorie_estimator.enhanced_estimator import EnhancedCalorieEstimator
        estimator = EnhancedCalorieEstimator()
        print("✅ Enhanced AI estimator initialized successfully!")
        print("🚀 Ready to analyze food photos with GPT-4o-mini + USDA data!")
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        print("Please check your API key and try again.")
    
    print("\n🎯 Next steps:")
    print("1. Run: python app.py")
    print("2. Visit: http://localhost:8000")
    print("3. Upload food photos and get AI-powered calorie estimates!")

def load_env_file():
    """Load environment variables from .env file"""
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        print(f"📁 Loaded environment variables from {env_file}")

if __name__ == "__main__":
    # Load existing .env file if it exists
    load_env_file()
    
    # Run setup
    setup_api_key()
