import sqlite3
import json
from datetime import datetime
from openai import OpenAI
from application.model import User, Pizza
from flask import session
from sqlalchemy import func

DEFAULT_MODEL = "gpt-4o-mini"  # You can change this to any OpenAI model available


def openai_chat(prompt: str, api_key: str) -> str:
    """Wrapper around OpenAI chat completion with explicit API key"""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # keep output stable
    )
    
    
    return response.choices[0].message.content.strip()


def extract_order(order_text: str, api_key: str):
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
    return json.loads(openai_chat(prompt, api_key))


def place_order(order_text: str, api_key: str):
    conn = None
    try:
        #  Extract order details with LLM
        order_info = extract_order(order_text, api_key)
        username = order_info["username"]
        pizza_name = order_info["pizza"]
        quantity = int(order_info["quantity"])
        
        
        # lookup User 
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User.query.get_or_404(session.get('user_id'))    
        
        # lookup Pizza
        pizza = Pizza.query.filter(func.lower(Pizza.name).contains(pizza_name.lower())).first()
        if not pizza:
            return f"❌ Sorry, we don’t have {pizza_name} on the menu."

      # Connect DB
        conn = sqlite3.connect("instance/pizza_shop.db")
        cursor = conn.cursor()

        total_price = pizza.price * quantity
        cursor.execute("""
            INSERT INTO "order" (user_id, pizza_id, quantity, total_price, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user.id, pizza.id, quantity, total_price, datetime.utcnow()))
        conn.commit()
        # confirmation from LLM
        confirm_prompt = f"""
        A user has successfully placed a pizza order.

        - User: {username}
        - Pizza: {pizza_name}
        - Quantity: {quantity}
        - Unit Price: {pizza.price}
        - Total Price: {total_price}

        Respond to the user in a friendly way confirming their order.
        """
        return openai_chat(confirm_prompt, api_key)

    except Exception as e:
        return f"❌ Error placing order: {e}"

    finally:
        if conn:
            conn.close()
