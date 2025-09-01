import sqlite3
from application.model import Pizza, Order, User
import json
from datetime import datetime
from flask import session
import requests
from sqlalchemy import func




# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:1b"

def chat_with_ollama(user_message, model_name=DEFAULT_MODEL):
    """Send a message to Ollama and get the response"""
    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "keep_alive": -1,
        }
        
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result['message']['content']
        else:
            return f"Error: Unable to connect to Ollama (status: {response.status_code})"
            
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Please ensure Ollama is running on localhost:11434"
    except Exception as e:
        return f"Error communicating with Ollama: {str(e)}"

def extract_order(order_text: str):
    """Use LLM to parse order into structured JSON"""
    prompt = f"""
    Extract structured information from this pizza order.
    Required fields: username, pizza, quantity.
    Return ONLY JSON.

    Example:
    Input: "My name is Bob and I want 3 margherita pizzas"
    Output: {{"username": "Bob", "pizza": "margherita", "quantity": 3}}

    Input: "{order_text}"
    """
    
    response_text = chat_with_ollama(prompt)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON from response if it contains extra text
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            raise ValueError(f"Could not parse JSON from response: {response_text}")


def place_order(order_text: str):
    # extract order details
    conn = sqlite3.connect('instance/pizza_shop.db')
    cursor = conn.cursor()

    order_info = extract_order(order_text)

    username = order_info["username"]
    pizza_name = order_info["pizza"]
    quantity = int(order_info["quantity"])

    # Step 2: lookup User or create if not exists
    user = User.query.filter_by(username=username).first()
    if not user:
        user = User.query.get_or_404(session.get('user_id'))
    
    # Step 3: lookup Pizza
    pizza = Pizza.query.filter(func.lower(Pizza.name).contains(pizza_name.lower())).first()
    if not pizza:
        return f"❌ Sorry, we don’t have {pizza_name} on the menu."

    # Step 4: create Order
    total_price = pizza.price * quantity
    cursor.execute("""
            INSERT INTO "order" (user_id, pizza_id, quantity, total_price, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user.id, pizza.id, quantity, total_price, datetime.utcnow()))
    conn.commit()
        
    # Step 5: ask LLM to confirm order in natural language
    confirm_prompt = f"""
    A user has successfully placed a pizza order.

    - User: {username}
    - Pizza: {pizza.name}
    - Quantity: {quantity}
    - Unit Price: ${pizza.price}
    - Total Price: ${total_price}

    Respond to the user in a friendly way confirming their order.
    """
    
    confirmation_response = chat_with_ollama(confirm_prompt)
    
    conn.close()
    return confirmation_response
