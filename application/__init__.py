from flask import Flask
from config import Config
from flask_sqlalchemy import SQLAlchemy


# Initialize Flask app
app = Flask(__name__)

app.config.from_object(Config)

db = SQLAlchemy(app)
from application import route