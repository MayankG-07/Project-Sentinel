import os
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the encryption key from the environment
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY not found in .env file. Please run generate_key.py and add it.")

# Initialize Fernet with the key
fernet = Fernet(ENCRYPTION_KEY.encode())

# Read the plaintext user data from users.json
try:
    with open("users.json", "r") as f:
        users_data = json.load(f)
except FileNotFoundError:
    print("Error: users.json not found. Please create this file with your user data.")
    exit(1)

# Convert the user data to a JSON string, then encode to bytes
users_bytes = json.dumps(users_data).encode()

# Encrypt the data
encrypted_users = fernet.encrypt(users_bytes)

# Write the encrypted data to users.json.enc
with open("users.json.enc", "wb") as f:
    f.write(encrypted_users)

print("Successfully encrypted users.json and created users.json.enc")
