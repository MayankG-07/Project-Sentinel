# generate_hash.py
import sys
from passlib.context import CryptContext

# This script generates a bcrypt hash for a given password.
# Usage: python generate_hash.py YOUR_PASSWORD_HERE

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_hash.py <password>")
        sys.exit(1)

    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    password = sys.argv[1]
    
    print(f"Generating hash for password: '{password}'")
    hashed_password = pwd_context.hash(password)
    print("SUCCESS! Copy the following hash:")
    print(hashed_password)
