from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy
#to increase response time in all pages
import os
os.environ['OLLAMA_KEEP_ALIVE'] = '-1'

# Initialize Flask app
app = Flask(__name__)

app.config.from_object(Config)
db = SQLAlchemy(app)

# Import routes at the end to avoid circular imports
from application import route  # noqa: F401 - imported for side effects (route registration)
