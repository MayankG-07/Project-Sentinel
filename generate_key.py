from cryptography.fernet import Fernet

# Generate a new key
key = Fernet.generate_key()

# Print the key to the console.
# Copy this key and add it to your .env file as ENCRYPTION_KEY
print("Generated Encryption Key:")
print(key.decode())
