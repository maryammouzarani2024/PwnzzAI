import sqlite3
from application.model import Pizza, Order, User
import json
from datetime import datetime
from flask import session
import requests
from sqlalchemy import func




# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b"

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
    If no username is mentioned, omit the username field or set it to null.
    Return ONLY JSON.
    Example pizzas are margherita, Hawaiian, pepperoni, BBQ chicken,veggie supreme, etc. 
   
    Example prompts:
    Input: "My name is Bob and I want 3 margherita pizzas "
    Output: {{"username": "Bob", "pizza": "margherita", "quantity": 3}}

    
    Input: "I want 2 pepperoni pizzas"
    Output: {{"pizza": "pepperoni", "quantity": 2}}

    Input: "{order_text}"
    Output:"""
    
    response_text = chat_with_ollama(prompt)

    try:
        # Clean the response text - remove any markdown formatting
        cleaned_text = response_text.strip()
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()
        
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        # Try to extract JSON from response if it contains extra text
        import re
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
       

def place_order(order_text: str):
    # extract order details
    print("order text: ", order_text)
    conn = sqlite3.connect('instance/pizza_shop.db')
    cursor = conn.cursor()

    order_info = extract_order(order_text)
    print("order info: ")
    print(order_info)
    
    if order_info:
        username = order_info.get("username")
        pizza_name = order_info["pizza"]
        quantity = int(order_info["quantity"])
    else: 
        return "❌ There was an error please try again."

    # Step 2: lookup User - if no username mentioned, use session user
    if username and username.lower() != "anonymous":
        user = User.query.filter_by(username=func.lower(username)).first()
        if not user:
            conn.close()
            return f"❌ User '{username}' not found. Please use an existing username or log in."
    else:
        # No username mentioned, use logged-in user
        user_id = session.get('user_id')
        if not user_id:
            conn.close()
            return "❌ You must be logged in to place an order. Please log in first."
        user = User.query.get(user_id)
        if not user:
            conn.close()
            return "❌ Invalid session. Please log in again."
        username=user.username
    # Step 3: lookup Pizza
    pizza = Pizza.query.filter(func.lower(Pizza.name).contains(pizza_name.lower())).first()
    if not pizza:
        return f"❌ Sorry, we don’t have {pizza_name} on the menu."

    print(username, pizza.name, quantity)
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
