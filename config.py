import os


class Config(object):
    
    # Configure SQLite database
    SQLALCHEMY_DATABASE_URI = 'sqlite:///pizza_shop.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False