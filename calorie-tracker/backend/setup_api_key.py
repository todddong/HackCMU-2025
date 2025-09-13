#!/usr/bin/env python3
"""
Setup script for OpenAI API key configuration
This script helps users set up their OpenAI API key for enhanced AI calorie estimation.
"""

import os
import sys
from pathlib import Path

def setup_api_key():
    """Interactive setup for OpenAI API key"""
    print("🤖 Enhanced AI Calorie Estimation Setup")
    print("=" * 50)
    print()
    print("This will set up your OpenAI API key for enhanced AI-powered calorie estimation.")
    print("The enhanced estimator uses GPT-4o-mini Vision API combined with USDA nutritional data.")
    print()
    print("To get your API key:")
    print("1. Visit: https://platform.openai.com/api-keys")
    print("2. Sign in to your OpenAI account")
    print("3. Click 'Create new secret key'")
    print("4. Copy the key (it starts with 'sk-')")
    print()
    
    # Check if key already exists
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            content = f.read()
            if "OPENAI_API_KEY" in content:
                print("✅ OpenAI API key already configured in .env file")
                return
    
    # Get API key from user
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    
    if not api_key:
        print("⚠️  Skipping API key setup. The app will use mock calorie estimation.")
        print("   You can set up the API key later by creating a .env file with:")
        print("   OPENAI_API_KEY=your_api_key_here")
        return
    
    if not api_key.startswith("sk-"):
        print("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
        return
    
    # Save to .env file
    try:
        with open(".env", "w") as f:
            f.write(f"OPENAI_API_KEY={api_key}\n")
        
        print("✅ API key saved successfully!")
        print("🚀 Enhanced AI calorie estimation is now enabled!")
        print()
        print("Features you'll get:")
        print("• GPT-4o-mini Vision for food identification")
        print("• USDA nutritional database integration")
        print("• Accurate calorie estimation from photos")
        print("• Detailed food descriptions")
        
    except Exception as e:
        print(f"❌ Error saving API key: {e}")
        print("You can manually create a .env file with:")
        print(f"OPENAI_API_KEY={api_key}")

def test_api_key():
    """Test if the API key works"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ No API key found")
            return False
        
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Test with a simple request
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        
        print("✅ API key is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_api_key()
    else:
        setup_api_key()

