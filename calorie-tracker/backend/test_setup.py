#!/usr/bin/env python3
"""
Test script to verify the calorie tracker setup is working correctly.
Run this script to check if all dependencies are installed and the app can start.
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported successfully")
    except ImportError as e:
        print(f"❌ FastAPI import failed: {e}")
        return False
    
    try:
        import sqlalchemy
        print("✅ SQLAlchemy imported successfully")
    except ImportError as e:
        print(f"❌ SQLAlchemy import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    try:
        from calorie_estimator.openai_estimator import OpenAICalorieEstimator
        print("✅ OpenAI Calorie Estimator imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI Calorie Estimator import failed: {e}")
        return False
    
    try:
        from calorie_estimator.mock_estimator import MockCalorieEstimator
        print("✅ Mock Calorie Estimator imported successfully")
    except ImportError as e:
        print(f"❌ Mock Calorie Estimator import failed: {e}")
        return False
    
    return True

def test_openai_setup():
    """Test OpenAI API key setup"""
    print("\nTesting OpenAI setup...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OPENAI_API_KEY environment variable is set")
        try:
            from calorie_estimator.openai_estimator import OpenAICalorieEstimator
            estimator = OpenAICalorieEstimator()
            print("✅ OpenAI Calorie Estimator initialized successfully")
            return True
        except Exception as e:
            print(f"❌ OpenAI Calorie Estimator initialization failed: {e}")
            return False
    else:
        print("⚠️  OPENAI_API_KEY environment variable not set")
        print("   The app will fall back to mock estimator")
        print("   To use real AI, set: export OPENAI_API_KEY='your_key_here'")
        return True

def test_database_setup():
    """Test database setup"""
    print("\nTesting database setup...")
    
    try:
        import models
        import database
        print("✅ Database models imported successfully")
        
        # Try to create tables
        models.Base.metadata.create_all(bind=database.engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Calorie Tracker Setup Test")
    print("=" * 40)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test OpenAI setup
    if not test_openai_setup():
        all_passed = False
    
    # Test database setup
    if not test_database_setup():
        all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! The calorie tracker is ready to run.")
        print("\nTo start the application:")
        print("  python app.py")
        print("\nThen visit: http://localhost:8000")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTo install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
