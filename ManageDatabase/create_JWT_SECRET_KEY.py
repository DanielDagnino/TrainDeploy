import secrets

# Generate a random 256-bit (32-byte) key
jwt_secret_key = secrets.token_hex(32)
print("Generated JWT_SECRET_KEY:", jwt_secret_key)
