import os


class Config(object):
    
    # Configure SQLite database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///pizza_shop.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    OLLAMA_MODEL_NAME="mistral:7b"