#!/usr/bin/env python3
"""
Database migration script to add user_id column to comment table
"""
import sqlite3
import os

def migrate_database():
    db_path = 'instance/pizza_shop.db'
    
    if not os.path.exists(db_path):
        print(f"Database file {db_path} not found!")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if user_id column already exists
        cursor.execute("PRAGMA table_info(comment)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'user_id' not in columns:
            print("Adding user_id column to comment table...")
            cursor.execute("ALTER TABLE comment ADD COLUMN user_id INTEGER REFERENCES user(id)")
            conn.commit()
            print("Migration completed successfully!")
        else:
            print("user_id column already exists in comment table.")
        
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return False

if __name__ == '__main__':
    migrate_database()