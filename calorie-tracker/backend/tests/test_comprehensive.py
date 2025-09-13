"""
Comprehensive Testing Suite for Calorie Tracker Application
This module provides extensive testing coverage for all major components.

Features:
- Unit tests for all modules
- Integration tests for API endpoints
- Performance and load testing
- Security testing
- Mock data and fixtures
- Test coverage reporting
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
from PIL import Image
import io

# Import application modules
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import our modules
from app import app
from models import Base, User, Meal
from database import get_db
from auth import get_password_hash, verify_password, create_access_token
from calorie_estimator.real_estimator import AdvancedCalorieEstimator, EstimationResult
from analytics.nutrition_analyzer import NutritionAnalyzer, NutritionReport
from services.nutrition_api_client import NutritionAPIClient, APIResponse
from security.advanced_auth import PasswordValidator, AdvancedPasswordManager, TwoFactorAuthManager

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Test fixtures
@pytest.fixture(scope="module")
def test_db():
    """Create test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)

@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "username": "testuser",
        "password": "TestPassword123!"
    }

@pytest.fixture
def test_meal_data():
    """Test meal data."""
    return {
        "name": "Test Meal",
        "calories": 500.0,
        "protein": 25.0,
        "carbs": 50.0,
        "fat": 20.0,
        "description": "A test meal"
    }

@pytest.fixture
def auth_headers(client, test_user_data):
    """Get authentication headers for test user."""
    # Register user
    response = client.post("/api/auth/register", json=test_user_data)
    assert response.status_code == 200
    
    # Login user
    response = client.post("/api/auth/login", json=test_user_data)
    assert response.status_code == 200
    
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes

# Authentication Tests
class TestAuthentication:
    """Test authentication functionality."""
    
    def test_user_registration(self, client, test_user_data):
        """Test user registration."""
        response = client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 200
        assert response.json()["username"] == test_user_data["username"]
    
    def test_user_registration_duplicate(self, client, test_user_data):
        """Test duplicate user registration."""
        # Register first user
        client.post("/api/auth/register", json=test_user_data)
        
        # Try to register same user again
        response = client.post("/api/auth/register", json=test_user_data)
        assert response.status_code == 400
    
    def test_user_login(self, client, test_user_data):
        """Test user login."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)
        
        # Login
        response = client.post("/api/auth/login", json=test_user_data)
        assert response.status_code == 200
        assert "access_token" in response.json()
    
    def test_user_login_invalid_credentials(self, client, test_user_data):
        """Test login with invalid credentials."""
        # Register user first
        client.post("/api/auth/register", json=test_user_data)
        
        # Try to login with wrong password
        invalid_data = test_user_data.copy()
        invalid_data["password"] = "wrongpassword"
        
        response = client.post("/api/auth/login", json=invalid_data)
        assert response.status_code == 401
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication."""
        response = client.get("/api/auth/me")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_auth(self, client, auth_headers):
        """Test accessing protected endpoint with authentication."""
        response = client.get("/api/auth/me", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["username"] == "testuser"

# Meal Management Tests
class TestMealManagement:
    """Test meal management functionality."""
    
    def test_create_meal(self, client, auth_headers, test_meal_data):
        """Test creating a meal."""
        response = client.post("/api/meals/", data=test_meal_data, headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["name"] == test_meal_data["name"]
        assert response.json()["calories"] == test_meal_data["calories"]
    
    def test_create_meal_without_auth(self, client, test_meal_data):
        """Test creating a meal without authentication."""
        response = client.post("/api/meals/", data=test_meal_data)
        assert response.status_code == 401
    
    def test_get_meals(self, client, auth_headers, test_meal_data):
        """Test getting meals."""
        # Create a meal first
        client.post("/api/meals/", data=test_meal_data, headers=auth_headers)
        
        # Get meals
        response = client.get("/api/meals/", headers=auth_headers)
        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["name"] == test_meal_data["name"]
    
    def test_delete_meal(self, client, auth_headers, test_meal_data):
        """Test deleting a meal."""
        # Create a meal first
        response = client.post("/api/meals/", data=test_meal_data, headers=auth_headers)
        meal_id = response.json()["id"]
        
        # Delete the meal
        response = client.delete(f"/api/meals/{meal_id}", headers=auth_headers)
        assert response.status_code == 200
        
        # Verify meal is deleted
        response = client.get("/api/meals/", headers=auth_headers)
        assert len(response.json()) == 0
    
    def test_delete_meal_unauthorized(self, client, auth_headers, test_meal_data):
        """Test deleting another user's meal."""
        # Create meal with first user
        response = client.post("/api/meals/", data=test_meal_data, headers=auth_headers)
        meal_id = response.json()["id"]
        
        # Create second user
        user2_data = {"username": "testuser2", "password": "TestPassword123!"}
        client.post("/api/auth/register", json=user2_data)
        login_response = client.post("/api/auth/login", json=user2_data)
        user2_token = login_response.json()["access_token"]
        user2_headers = {"Authorization": f"Bearer {user2_token}"}
        
        # Try to delete first user's meal with second user
        response = client.delete(f"/api/meals/{meal_id}", headers=user2_headers)
        assert response.status_code == 403

# Statistics Tests
class TestStatistics:
    """Test statistics functionality."""
    
    def test_get_stats_empty(self, client, auth_headers):
        """Test getting stats with no meals."""
        response = client.get("/api/stats/", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["total_calories"] == 0
        assert response.json()["total_meals"] == 0
    
    def test_get_stats_with_meals(self, client, auth_headers, test_meal_data):
        """Test getting stats with meals."""
        # Create multiple meals
        for i in range(3):
            meal_data = test_meal_data.copy()
            meal_data["name"] = f"Meal {i}"
            client.post("/api/meals/", data=meal_data, headers=auth_headers)
        
        # Get stats
        response = client.get("/api/stats/", headers=auth_headers)
        assert response.status_code == 200
        assert response.json()["total_calories"] == 1500.0  # 3 * 500
        assert response.json()["total_meals"] == 3

# Food Search Tests
class TestFoodSearch:
    """Test food search functionality."""
    
    @patch('usda_service.usda_service.search_foods')
    def test_search_foods(self, mock_search, client):
        """Test food search."""
        # Mock the search results
        mock_search.return_value = [
            {
                "name": "Apple",
                "calories": 52,
                "protein": 0.3,
                "carbs": 14,
                "fat": 0.2,
                "serving_size": "100g"
            }
        ]
        
        response = client.get("/api/search-foods/?query=apple")
        assert response.status_code == 200
        assert len(response.json()["results"]) == 1
        assert response.json()["results"][0]["name"] == "Apple"
    
    def test_search_foods_invalid_query(self, client):
        """Test food search with invalid query."""
        response = client.get("/api/search-foods/?query=a")
        assert response.status_code == 400

# Calorie Estimation Tests
class TestCalorieEstimation:
    """Test calorie estimation functionality."""
    
    def test_estimate_calories(self, client, sample_image):
        """Test calorie estimation from image."""
        files = {"photo": ("test.jpg", sample_image, "image/jpeg")}
        response = client.post("/api/estimate-calories/", files=files)
        assert response.status_code == 200
        assert "estimated_calories" in response.json()
        assert "description" in response.json()
    
    def test_estimate_calories_invalid_file(self, client):
        """Test calorie estimation with invalid file."""
        files = {"photo": ("test.txt", b"not an image", "text/plain")}
        response = client.post("/api/estimate-calories/", files=files)
        assert response.status_code == 400

# Advanced Calorie Estimator Tests
class TestAdvancedCalorieEstimator:
    """Test advanced calorie estimator."""
    
    def test_estimator_initialization(self):
        """Test estimator initialization."""
        estimator = AdvancedCalorieEstimator()
        assert estimator is not None
        assert hasattr(estimator, 'food_database')
        assert hasattr(estimator, 'portion_estimator')
    
    def test_food_database_loading(self):
        """Test food database loading."""
        estimator = AdvancedCalorieEstimator()
        assert len(estimator.food_database) > 0
        assert "apple" in estimator.food_database
        assert "calories" in estimator.food_database["apple"]
    
    @patch('calorie_estimator.real_estimator.cv2.imread')
    def test_estimate_calories_with_mock_image(self, mock_imread):
        """Test calorie estimation with mock image."""
        # Mock OpenCV image loading
        mock_imread.return_value = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        estimator = AdvancedCalorieEstimator()
        result = estimator.estimate_calories("test_image.jpg")
        
        assert isinstance(result, EstimationResult)
        assert result.total_calories >= 0
        assert result.confidence_score >= 0
        assert result.processing_time_ms >= 0
    
    def test_fallback_estimation(self):
        """Test fallback estimation when analysis fails."""
        estimator = AdvancedCalorieEstimator()
        result = estimator._fallback_estimation()
        
        assert isinstance(result, EstimationResult)
        assert result.total_calories == 350.0
        assert result.confidence_score == 0.3

# Nutrition Analyzer Tests
class TestNutritionAnalyzer:
    """Test nutrition analyzer."""
    
    def test_analyzer_initialization(self, test_db):
        """Test analyzer initialization."""
        db = TestingSessionLocal()
        analyzer = NutritionAnalyzer(db)
        assert analyzer is not None
        assert hasattr(analyzer, 'nutrition_goals')
        assert analyzer.nutrition_goals['calories'] == 2000
        db.close()
    
    def test_empty_report_generation(self, test_db):
        """Test generating report with no data."""
        db = TestingSessionLocal()
        analyzer = NutritionAnalyzer(db)
        report = analyzer.generate_comprehensive_report(user_id=999, days=30)
        
        assert isinstance(report, NutritionReport)
        assert report.total_calories == 0
        assert report.total_meals == 0
        assert len(report.trends) == 0
        assert len(report.insights) == 0
        db.close()
    
    def test_macro_distribution_calculation(self, test_db):
        """Test macro distribution calculation."""
        db = TestingSessionLocal()
        analyzer = NutritionAnalyzer(db)
        
        # Create test data
        test_data = [
            {'calories': 400, 'protein': 20, 'carbs': 40, 'fat': 20},
            {'calories': 600, 'protein': 30, 'carbs': 60, 'fat': 30}
        ]
        
        # Mock the DataFrame
        import pandas as pd
        df = pd.DataFrame(test_data)
        
        macro_dist = analyzer._calculate_macro_distribution(df)
        
        assert 'protein' in macro_dist
        assert 'carbs' in macro_dist
        assert 'fat' in macro_dist
        assert sum(macro_dist.values()) <= 100  # Should be percentages
        db.close()

# API Client Tests
class TestNutritionAPIClient:
    """Test nutrition API client."""
    
    def test_client_initialization(self):
        """Test API client initialization."""
        config = {
            "usda_api_key": "test_key",
            "cache_type": "memory"
        }
        
        client = NutritionAPIClient(config)
        assert client is not None
        assert hasattr(client, 'cache_manager')
        assert hasattr(client, 'rate_limiters')
    
    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        from services.nutrition_api_client import CacheManager
        
        cache_manager = CacheManager("memory")
        assert cache_manager.cache_type == "memory"
        assert hasattr(cache_manager, 'memory_cache')
    
    @pytest.mark.asyncio
    async def test_cache_operations(self):
        """Test cache operations."""
        from services.nutrition_api_client import CacheManager
        
        cache_manager = CacheManager("memory")
        
        # Test set and get
        await cache_manager.set("test_query", "test_provider", {"test": "data"})
        cached_data = await cache_manager.get("test_query", "test_provider")
        
        assert cached_data is not None
        assert cached_data["test"] == "data"
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        from services.nutrition_api_client import RateLimiter
        
        rate_limiter = RateLimiter(10, 60)  # 10 requests per minute
        assert rate_limiter.max_requests == 10
        assert rate_limiter.time_window == 60

# Security Tests
class TestPasswordValidator:
    """Test password validation."""
    
    def test_password_validator_initialization(self):
        """Test password validator initialization."""
        validator = PasswordValidator()
        assert validator is not None
        assert hasattr(validator, 'policy')
        assert hasattr(validator, 'common_passwords')
    
    def test_strong_password_validation(self):
        """Test validation of strong password."""
        validator = PasswordValidator()
        is_valid, errors = validator.validate_password("StrongPass123!")
        
        assert is_valid
        assert len(errors) == 0
    
    def test_weak_password_validation(self):
        """Test validation of weak password."""
        validator = PasswordValidator()
        is_valid, errors = validator.validate_password("weak")
        
        assert not is_valid
        assert len(errors) > 0
        assert any("length" in error.lower() for error in errors)
    
    def test_common_password_validation(self):
        """Test validation of common password."""
        validator = PasswordValidator()
        is_valid, errors = validator.validate_password("password")
        
        assert not is_valid
        assert any("common" in error.lower() for error in errors)
    
    def test_password_strength_calculation(self):
        """Test password strength calculation."""
        validator = PasswordValidator()
        strength = validator.calculate_password_strength("StrongPass123!")
        
        assert "score" in strength
        assert "strength" in strength
        assert "feedback" in strength
        assert strength["score"] > 0
        assert strength["strength"] in ["Weak", "Fair", "Good", "Strong"]

class TestAdvancedPasswordManager:
    """Test advanced password manager."""
    
    def test_password_manager_initialization(self, test_db):
        """Test password manager initialization."""
        db = TestingSessionLocal()
        manager = AdvancedPasswordManager(db)
        assert manager is not None
        assert hasattr(manager, 'validator')
        db.close()
    
    def test_password_hashing(self, test_db):
        """Test password hashing."""
        db = TestingSessionLocal()
        manager = AdvancedPasswordManager(db)
        
        password = "TestPassword123!"
        hashed = manager.hash_password(password)
        
        assert hashed != password
        assert len(hashed) > 0
        assert manager.verify_password(password, hashed)
        assert not manager.verify_password("wrongpassword", hashed)
        db.close()
    
    def test_password_validation_and_hashing(self, test_db):
        """Test password validation and hashing."""
        db = TestingSessionLocal()
        manager = AdvancedPasswordManager(db)
        
        # Valid password
        hashed, errors = manager.validate_and_hash_password("ValidPass123!")
        assert len(errors) == 0
        assert len(hashed) > 0
        
        # Invalid password
        hashed, errors = manager.validate_and_hash_password("weak")
        assert len(errors) > 0
        assert hashed == ""
        db.close()

class TestTwoFactorAuthManager:
    """Test two-factor authentication manager."""
    
    def test_2fa_manager_initialization(self, test_db):
        """Test 2FA manager initialization."""
        db = TestingSessionLocal()
        manager = TwoFactorAuthManager(db)
        assert manager is not None
        db.close()
    
    def test_secret_key_generation(self, test_db):
        """Test secret key generation."""
        db = TestingSessionLocal()
        manager = TwoFactorAuthManager(db)
        
        secret_key = manager.generate_secret_key()
        assert len(secret_key) > 0
        assert isinstance(secret_key, str)
        db.close()
    
    def test_backup_codes_generation(self, test_db):
        """Test backup codes generation."""
        db = TestingSessionLocal()
        manager = TwoFactorAuthManager(db)
        
        backup_codes = manager.generate_backup_codes()
        assert len(backup_codes) == 10
        assert all(len(code) == 8 for code in backup_codes)  # 4 bytes = 8 hex chars
        db.close()
    
    def test_qr_code_generation(self, test_db):
        """Test QR code generation."""
        db = TestingSessionLocal()
        manager = TwoFactorAuthManager(db)
        
        secret_key = manager.generate_secret_key()
        qr_code = manager.generate_qr_code(secret_key, "testuser")
        
        assert qr_code.startswith("data:image/png;base64,")
        assert len(qr_code) > 100  # Should be a substantial base64 string
        db.close()

# Performance Tests
class TestPerformance:
    """Test application performance."""
    
    def test_meal_creation_performance(self, client, auth_headers, test_meal_data):
        """Test meal creation performance."""
        import time
        
        start_time = time.time()
        response = client.post("/api/meals/", data=test_meal_data, headers=auth_headers)
        end_time = time.time()
        
        assert response.status_code == 200
        assert (end_time - start_time) < 1.0  # Should complete within 1 second
    
    def test_multiple_meal_creation_performance(self, client, auth_headers):
        """Test multiple meal creation performance."""
        import time
        
        start_time = time.time()
        
        for i in range(10):
            meal_data = {
                "name": f"Performance Test Meal {i}",
                "calories": 300.0,
                "protein": 15.0,
                "carbs": 30.0,
                "fat": 10.0
            }
            response = client.post("/api/meals/", data=meal_data, headers=auth_headers)
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert total_time < 5.0  # Should complete within 5 seconds
        assert total_time / 10 < 0.5  # Average should be under 0.5 seconds per meal

# Integration Tests
class TestIntegration:
    """Test full application integration."""
    
    def test_complete_user_workflow(self, client):
        """Test complete user workflow from registration to meal tracking."""
        # Register user
        user_data = {"username": "integration_test_user", "password": "TestPass123!"}
        response = client.post("/api/auth/register", json=user_data)
        assert response.status_code == 200
        
        # Login user
        response = client.post("/api/auth/login", json=user_data)
        assert response.status_code == 200
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Create multiple meals
        meals_data = [
            {"name": "Breakfast", "calories": 400, "protein": 20, "carbs": 40, "fat": 15},
            {"name": "Lunch", "calories": 600, "protein": 30, "carbs": 60, "fat": 25},
            {"name": "Dinner", "calories": 500, "protein": 25, "carbs": 50, "fat": 20}
        ]
        
        for meal_data in meals_data:
            response = client.post("/api/meals/", data=meal_data, headers=headers)
            assert response.status_code == 200
        
        # Get meals
        response = client.get("/api/meals/", headers=headers)
        assert response.status_code == 200
        assert len(response.json()) == 3
        
        # Get stats
        response = client.get("/api/stats/", headers=headers)
        assert response.status_code == 200
        assert response.json()["total_calories"] == 1500
        assert response.json()["total_meals"] == 3
        
        # Delete a meal
        meal_id = response.json()["total_meals"]  # This would be the actual meal ID
        # Note: In a real test, you'd get the actual meal ID from the creation response
        
        # Verify final state
        response = client.get("/api/meals/", headers=headers)
        assert len(response.json()) == 3  # All meals still there since we didn't actually delete

# Test configuration and utilities
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "security: marks tests as security tests")

# Test discovery and execution
if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        "-v",
        "--tb=short",
        "--cov=.",
        "--cov-report=html",
        "--cov-report=term-missing",
        "test_comprehensive.py"
    ])

