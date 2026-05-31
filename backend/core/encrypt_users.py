
from pathlib import Path
import os
import json
from cryptography.fernet import Fernet
from dotenv import load_dotenv

# --- DYNAMIC PATH CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "backend" / "data"

# Load environment variables from .env file in the project root
load_dotenv(BASE_DIR / ".env")

# --- ENCRYPTION SETUP ---
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY not found in .env file. Please run generate_key.py and add it.")

# Initialize Fernet with the key
fernet = Fernet(ENCRYPTION_KEY.encode())

# Define file paths
users_json_path = DATA_DIR / "users.json"
encrypted_users_path = DATA_DIR / "users.json.enc"

# Read the plaintext user data from users.json
try:
    with open(users_json_path, "r") as f:
        users_data = json.load(f)
except FileNotFoundError:
    print(f"Error: {users_json_path} not found. Please create this file with your user data.")
    exit(1)

# Convert the user data to a JSON string, then encode to bytes
users_bytes = json.dumps(users_data).encode()

# Encrypt the data
encrypted_users = fernet.encrypt(users_bytes)

# Write the encrypted data to users.json.enc
with open(encrypted_users_path, "wb") as f:
    f.write(encrypted_users)

print(f"Successfully encrypted {users_json_path} and created {encrypted_users_path}")
