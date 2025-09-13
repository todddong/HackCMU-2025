"""
Advanced Security and Authentication Module
This module provides enterprise-grade security features for the calorie tracker application.

Features:
- Advanced password policies and validation
- OAuth 2.0 integration (Google, GitHub, etc.)
- Two-factor authentication (2FA)
- Session management and security
- Rate limiting and brute force protection
- Security logging and monitoring
- Data encryption and secure storage
"""

import hashlib
import secrets
import pyotp
import qrcode
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import re
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
from sqlalchemy.orm import Session
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import aiohttp
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SecurityEvent:
    """Represents a security event for logging."""
    event_type: str
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    details: Dict[str, Any]
    success: bool

@dataclass
class PasswordPolicy:
    """Password policy configuration."""
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_numbers: bool = True
    require_special_chars: bool = True
    min_special_chars: int = 1
    forbidden_patterns: List[str] = None
    max_consecutive_chars: int = 3
    password_history_count: int = 5

@dataclass
class TwoFactorAuth:
    """Two-factor authentication configuration."""
    secret_key: str
    backup_codes: List[str]
    is_enabled: bool
    last_used: Optional[datetime]

class PasswordValidator:
    """Advanced password validation with comprehensive security checks."""
    
    def __init__(self, policy: PasswordPolicy = None):
        self.policy = policy or PasswordPolicy()
        self.common_passwords = self._load_common_passwords()
    
    def _load_common_passwords(self) -> set:
        """Load list of common passwords for validation."""
        # In production, load from a comprehensive database
        return {
            "password", "123456", "password123", "admin", "qwerty",
            "letmein", "welcome", "monkey", "1234567890", "abc123"
        }
    
    def validate_password(self, password: str, username: str = None) -> Tuple[bool, List[str]]:
        """
        Validate password against security policy.
        
        Args:
            password: Password to validate
            username: Username for context checks
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Length validation
        if len(password) < self.policy.min_length:
            errors.append(f"Password must be at least {self.policy.min_length} characters long")
        
        if len(password) > self.policy.max_length:
            errors.append(f"Password must be no more than {self.policy.max_length} characters long")
        
        # Character requirements
        if self.policy.require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.policy.require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.policy.require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        if self.policy.require_special_chars:
            special_chars = re.findall(r'[!@#$%^&*(),.?":{}|<>]', password)
            if len(special_chars) < self.policy.min_special_chars:
                errors.append(f"Password must contain at least {self.policy.min_special_chars} special character(s)")
        
        # Consecutive character check
        if self._has_consecutive_chars(password, self.policy.max_consecutive_chars):
            errors.append(f"Password cannot have more than {self.policy.max_consecutive_chars} consecutive identical characters")
        
        # Common password check
        if password.lower() in self.common_passwords:
            errors.append("Password is too common and easily guessable")
        
        # Username similarity check
        if username and self._is_similar_to_username(password, username):
            errors.append("Password cannot be similar to username")
        
        # Forbidden patterns
        if self.policy.forbidden_patterns:
            for pattern in self.policy.forbidden_patterns:
                if re.search(pattern, password, re.IGNORECASE):
                    errors.append(f"Password contains forbidden pattern: {pattern}")
        
        return len(errors) == 0, errors
    
    def _has_consecutive_chars(self, password: str, max_consecutive: int) -> bool:
        """Check for consecutive identical characters."""
        count = 1
        for i in range(1, len(password)):
            if password[i] == password[i-1]:
                count += 1
                if count > max_consecutive:
                    return True
            else:
                count = 1
        return False
    
    def _is_similar_to_username(self, password: str, username: str) -> bool:
        """Check if password is too similar to username."""
        password_lower = password.lower()
        username_lower = username.lower()
        
        # Check if password contains username
        if username_lower in password_lower:
            return True
        
        # Check if username contains password (for short passwords)
        if len(password) < 6 and password_lower in username_lower:
            return True
        
        return False
    
    def calculate_password_strength(self, password: str) -> Dict[str, Any]:
        """Calculate password strength score and feedback."""
        score = 0
        feedback = []
        
        # Length score
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1
        if len(password) >= 16:
            score += 1
        
        # Character variety score
        if re.search(r'[a-z]', password):
            score += 1
        if re.search(r'[A-Z]', password):
            score += 1
        if re.search(r'\d', password):
            score += 1
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            score += 1
        
        # Complexity bonus
        unique_chars = len(set(password))
        if unique_chars > len(password) * 0.6:
            score += 1
        
        # Determine strength level
        if score <= 2:
            strength = "Weak"
        elif score <= 4:
            strength = "Fair"
        elif score <= 6:
            strength = "Good"
        else:
            strength = "Strong"
        
        # Generate feedback
        if len(password) < 8:
            feedback.append("Use at least 8 characters")
        if not re.search(r'[A-Z]', password):
            feedback.append("Add uppercase letters")
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            feedback.append("Add special characters")
        
        return {
            "score": score,
            "max_score": 7,
            "strength": strength,
            "feedback": feedback
        }

class AdvancedPasswordManager:
    """Advanced password management with secure hashing and history."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.validator = PasswordValidator()
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt with salt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def validate_and_hash_password(self, password: str, username: str = None) -> Tuple[str, List[str]]:
        """Validate password and return hash if valid."""
        is_valid, errors = self.validator.validate_password(password, username)
        
        if not is_valid:
            return "", errors
        
        hashed = self.hash_password(password)
        return hashed, []
    
    def check_password_history(self, user_id: int, new_password: str) -> bool:
        """Check if password was used recently."""
        # This would query the database for password history
        # For now, return True (password is new)
        return True

class TwoFactorAuthManager:
    """Two-factor authentication manager with TOTP support."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def generate_secret_key(self) -> str:
        """Generate a new TOTP secret key."""
        return pyotp.random_base32()
    
    def generate_qr_code(self, secret_key: str, username: str) -> str:
        """Generate QR code for TOTP setup."""
        totp_uri = pyotp.totp.TOTP(secret_key).provisioning_uri(
            name=username,
            issuer_name="Calorie Tracker"
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64 string
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_str}"
    
    def generate_backup_codes(self, count: int = 10) -> List[str]:
        """Generate backup codes for 2FA."""
        return [secrets.token_hex(4).upper() for _ in range(count)]
    
    def verify_totp_code(self, secret_key: str, code: str) -> bool:
        """Verify TOTP code."""
        totp = pyotp.TOTP(secret_key)
        return totp.verify(code, valid_window=1)
    
    def verify_backup_code(self, user_id: int, code: str) -> bool:
        """Verify backup code."""
        # This would check against stored backup codes in database
        # For now, return True
        return True

class OAuthManager:
    """OAuth 2.0 integration manager."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.providers = {
            "google": {
                "client_id": "your_google_client_id",
                "client_secret": "your_google_client_secret",
                "auth_url": "https://accounts.google.com/o/oauth2/auth",
                "token_url": "https://oauth2.googleapis.com/token",
                "user_info_url": "https://www.googleapis.com/oauth2/v2/userinfo"
            },
            "github": {
                "client_id": "your_github_client_id",
                "client_secret": "your_github_client_secret",
                "auth_url": "https://github.com/login/oauth/authorize",
                "token_url": "https://github.com/login/oauth/access_token",
                "user_info_url": "https://api.github.com/user"
            }
        }
    
    async def get_authorization_url(self, provider: str, state: str) -> str:
        """Get OAuth authorization URL."""
        if provider not in self.providers:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.providers[provider]
        params = {
            "client_id": config["client_id"],
            "redirect_uri": f"http://localhost:8000/auth/{provider}/callback",
            "scope": "openid email profile",
            "response_type": "code",
            "state": state
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{config['auth_url']}?{query_string}"
    
    async def exchange_code_for_token(self, provider: str, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        if provider not in self.providers:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.providers[provider]
        
        data = {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "code": code,
            "redirect_uri": f"http://localhost:8000/auth/{provider}/callback"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(config["token_url"], data=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to exchange code for token"
                    )
    
    async def get_user_info(self, provider: str, access_token: str) -> Dict[str, Any]:
        """Get user information from OAuth provider."""
        if provider not in self.providers:
            raise ValueError(f"Unsupported OAuth provider: {provider}")
        
        config = self.providers[provider]
        headers = {"Authorization": f"Bearer {access_token}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(config["user_info_url"], headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Failed to get user information"
                    )

class SecurityMonitor:
    """Security monitoring and logging system."""
    
    def __init__(self, db_session: Session, redis_client: redis.Redis = None):
        self.db = db_session
        self.redis = redis_client
        self.failed_attempts = {}  # In production, use Redis
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event."""
        logger.info(f"Security Event: {event.event_type} - {event.severity} - {event.details}")
        
        # In production, store in database and send alerts for critical events
        if event.severity == "critical":
            self._send_security_alert(event)
    
    def _send_security_alert(self, event: SecurityEvent) -> None:
        """Send security alert for critical events."""
        # In production, integrate with alerting system (email, Slack, etc.)
        logger.critical(f"CRITICAL SECURITY EVENT: {event.event_type} - {event.details}")
    
    def check_brute_force_attempts(self, ip_address: str, user_id: int = None) -> bool:
        """Check for brute force attempts."""
        key = f"failed_attempts:{ip_address}:{user_id or 'anonymous'}"
        
        if self.redis:
            attempts = self.redis.get(key)
            if attempts and int(attempts) > 5:  # More than 5 failed attempts
                return True
        else:
            # Fallback to in-memory tracking
            if key in self.failed_attempts:
                if self.failed_attempts[key] > 5:
                    return True
        
        return False
    
    def record_failed_attempt(self, ip_address: str, user_id: int = None) -> None:
        """Record failed login attempt."""
        key = f"failed_attempts:{ip_address}:{user_id or 'anonymous'}"
        
        if self.redis:
            self.redis.incr(key)
            self.redis.expire(key, 3600)  # Expire after 1 hour
        else:
            self.failed_attempts[key] = self.failed_attempts.get(key, 0) + 1
    
    def clear_failed_attempts(self, ip_address: str, user_id: int = None) -> None:
        """Clear failed attempts after successful login."""
        key = f"failed_attempts:{ip_address}:{user_id or 'anonymous'}"
        
        if self.redis:
            self.redis.delete(key)
        else:
            self.failed_attempts.pop(key, None)

class DataEncryption:
    """Data encryption utilities for sensitive information."""
    
    def __init__(self, master_key: str = None):
        self.master_key = master_key or Fernet.generate_key()
        self.cipher = Fernet(self.master_key)
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        return self.cipher.decrypt(encrypted_data.encode()).decode()
    
    def encrypt_user_data(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt user data fields."""
        encrypted_data = user_data.copy()
        
        # Encrypt sensitive fields
        sensitive_fields = ['email', 'phone', 'address']
        for field in sensitive_fields:
            if field in encrypted_data and encrypted_data[field]:
                encrypted_data[field] = self.encrypt_data(str(encrypted_data[field]))
        
        return encrypted_data
    
    def decrypt_user_data(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt user data fields."""
        decrypted_data = encrypted_data.copy()
        
        # Decrypt sensitive fields
        sensitive_fields = ['email', 'phone', 'address']
        for field in sensitive_fields:
            if field in decrypted_data and decrypted_data[field]:
                try:
                    decrypted_data[field] = self.decrypt_data(decrypted_data[field])
                except Exception as e:
                    logger.error(f"Failed to decrypt {field}: {e}")
        
        return decrypted_data

class AdvancedSecurityManager:
    """Main security manager that coordinates all security features."""
    
    def __init__(self, db_session: Session, redis_client: redis.Redis = None):
        self.db = db_session
        self.password_manager = AdvancedPasswordManager(db_session)
        self.two_factor_manager = TwoFactorAuthManager(db_session)
        self.oauth_manager = OAuthManager(db_session)
        self.security_monitor = SecurityMonitor(db_session, redis_client)
        self.data_encryption = DataEncryption()
    
    def validate_password_strength(self, password: str, username: str = None) -> Dict[str, Any]:
        """Validate password and return strength analysis."""
        is_valid, errors = self.password_manager.validator.validate_password(password, username)
        strength_analysis = self.password_manager.validator.calculate_password_strength(password)
        
        return {
            "is_valid": is_valid,
            "errors": errors,
            "strength": strength_analysis
        }
    
    def setup_two_factor_auth(self, user_id: int, username: str) -> Dict[str, Any]:
        """Setup two-factor authentication for user."""
        secret_key = self.two_factor_manager.generate_secret_key()
        qr_code = self.two_factor_manager.generate_qr_code(secret_key, username)
        backup_codes = self.two_factor_manager.generate_backup_codes()
        
        # In production, store secret_key and backup_codes in database
        
        return {
            "secret_key": secret_key,
            "qr_code": qr_code,
            "backup_codes": backup_codes
        }
    
    def verify_two_factor_auth(self, secret_key: str, code: str) -> bool:
        """Verify two-factor authentication code."""
        return self.two_factor_manager.verify_totp_code(secret_key, code)
    
    def log_security_event(self, event_type: str, user_id: int, request: Request, 
                          severity: str = "medium", details: Dict[str, Any] = None, 
                          success: bool = True) -> None:
        """Log security event with request context."""
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=request.client.host,
            user_agent=request.headers.get("user-agent", ""),
            timestamp=datetime.now(),
            severity=severity,
            details=details or {},
            success=success
        )
        
        self.security_monitor.log_security_event(event)
    
    def check_security_restrictions(self, request: Request, user_id: int = None) -> bool:
        """Check if request should be blocked due to security restrictions."""
        ip_address = request.client.host
        
        # Check for brute force attempts
        if self.security_monitor.check_brute_force_attempts(ip_address, user_id):
            self.log_security_event(
                "brute_force_blocked", 
                user_id, 
                request, 
                "high", 
                {"ip_address": ip_address},
                False
            )
            return False
        
        return True

# Example usage
if __name__ == "__main__":
    # This would be used with actual database and Redis connections
    # from database import get_db
    # import redis
    # 
    # db = next(get_db())
    # redis_client = redis.Redis(host='localhost', port=6379, db=0)
    # 
    # security_manager = AdvancedSecurityManager(db, redis_client)
    # 
    # # Validate password
    # result = security_manager.validate_password_strength("MySecure123!", "john_doe")
    # print(f"Password validation: {result}")
    # 
    # # Setup 2FA
    # two_fa_setup = security_manager.setup_two_factor_auth(1, "john_doe")
    # print(f"2FA setup: {two_fa_setup}")
    pass

