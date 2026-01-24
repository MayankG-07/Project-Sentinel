import os
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from passlib.context import CryptContext

# --- CONFIGURATION ---
# In a real application, this would come from a secure database.
# For now, we define it here. The keys are hashed for security.
# Plaintext keys: "admin_key"
FAKE_USERS_DB = {
    "admin": {
        "hashed_key": "$2b$12$HDgiFJQiPQ86AcphdnVVq.y9xwMsGGWcuLbddUO8X0gUlZkspC/tC", # admin_key
        "roles": ["admin", "finance", "legal"],
    }
}

# --- SECURITY SETUP ---
api_key_header = APIKeyHeader(name="X-API-Key")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- AUTHENTICATION ---
def get_user(api_key: str):
    """Finds a user by their API key."""
    for username, user_data in FAKE_USERS_DB.items():
        try:
            if pwd_context.verify(api_key, user_data["hashed_key"]):
                return {"username": username, **user_data}
        except Exception:
            # This will catch errors from malformed hashes and allow the loop to continue.
            continue
    return None

# --- AUTHORIZATION ---
def get_current_user(api_key: str = Security(api_key_header)):
    """
    FastAPI dependency to get the current user from an API key.
    Raises an HTTPException if the key is invalid.
    """
    user = get_user(api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return user

def has_role(required_role: str):
    """
    FastAPI dependency to check if the current user has the required role.
    """
    def role_checker(current_user: dict = Security(get_current_user)):
        if required_role not in current_user["roles"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to perform this action.",
            )
        return current_user
    return role_checker
