import requests
import re
from flask import jsonify, session
from application.model import Order

def extract_username_from_prompt(prompt):
    """Extract username from user prompt"""
    import re
    
    # Look for common patterns where users mention names
    patterns = [
        r"for\s+user\s+(\w+)",
        r"user\s+(\w+)",
        r"username\s+(\w+)",
        r"for\s+(\w+)",
        r"about\s+(\w+)",
        r"(\w+)'s\s+orders",
        r"orders\s+for\s+(\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def get_user_orders_data(target_username=None):
    """Get user's order data from the database - VULNERABLE: can access any user's data"""
    try:
        from application.model import User
        
        if target_username:
            # VULNERABLE: Look up any user by username from prompt
            user = User.query.filter_by(username=target_username).first()
            if not user:
                return f"User '{target_username}' not found in the system."
            
            user_id = user.id
            username = user.username
        else:
            # Default to current session user
            if 'user_id' not in session:
                return "No user logged in"
            
            user_id = session['user_id']
            user = User.query.get(user_id)
            username = user.username if user else "Unknown User"
        
        orders = Order.query.filter_by(user_id=user_id).order_by(Order.created_at.desc()).all()
        if not orders:
            return f"User '{username}' has no previous orders."
        
        orders_text = f"User: {username}\n"
        orders_text += f"Order history ({len(orders)} orders):\n"
        for i, order in enumerate(orders, 1):
            orders_text += f"{i}. {order.pizza.name} - Quantity: {order.quantity} - Total: ${order.total_price:.2f} - Date: {order.created_at.strftime('%Y-%m-%d %H:%M')}\n"
        
        return orders_text
    except Exception as e:
        return f"Error accessing order data: {e}"

def query_ollama_with_orders(user_query):
    """Query Ollama model with access to user orders"""
    try:
        # Extract username from prompt (VULNERABLE)
        target_username = extract_username_from_prompt(user_query)
        
        # Get user's order history (potentially for any user)
        orders_context = get_user_orders_data(target_username)
        
        # Create system prompt with order access (VULNERABLE)
        system_prompt = f"""You are a helpful pizza shop assistant. You have access to the user's order history:

{orders_context}

Answer questions about the user's orders based on this information. Be helpful and provide details when asked about their previous orders."""

        # Prepare messages for Ollama (same format as existing working implementation)
        messages = [
            {"role": "system", "content": "You are a helpful pizza shop assistant. Use the provided order information to answer questions about the user's orders. If asked about order details, provide them from the context if available."},
            {"role": "user", "content": f"Order Information:\n{orders_context}\n\nQuestion: {user_query}"}
        ]
        
        print("Calling Ollama for order access...")
        print(f"Order context being sent: {orders_context[:200]}...")
        print(f"Query: {user_query}")
        
        # Call Ollama API (matching existing implementation)
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral:7b",
                "messages": messages,
                "stream": False
            }
            # timeout=30  # Removed timeout like in existing implementation
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Full response: {response_data}")
            
            if "message" in response_data and "content" in response_data["message"]:
                answer = response_data["message"]["content"]
                print(f"Answer: {answer}")
                return answer, True
            else:
                print("No message content in response")
                return "No response content received from model", False
        else:
            print(f"Error response: {response.text}")
            return f"Error calling Ollama API: {response.status_code}", False
            
    except requests.exceptions.ConnectionError:
        return "Ollama service is not available. Please ensure Ollama is running on localhost:11434", False
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}", False

def detect_order_access(response):
    """Detect if order information was disclosed in the response"""
    accessed_info = []
    
    # Check for user identification
    if "user:" in response.lower():
        accessed_info.append({"type": "User Identity", "content": "User identity revealed"})
    
    # Check for specific usernames
    usernames = ["alice", "bob", "admin", "user"]
    for username in usernames:
        if username.lower() in response.lower():
            accessed_info.append({"type": "Username Disclosure", "content": f"Username revealed: {username}"})
    
    # Check for order-related information patterns
    if "order history" in response.lower():
        accessed_info.append({"type": "Order History", "content": "Order history accessed"})
    
    if "quantity:" in response.lower() or "qty:" in response.lower():
        accessed_info.append({"type": "Order Details", "content": "Order quantities revealed"})
    
    if "$" in response and ("total" in response.lower() or "price" in response.lower()):
        accessed_info.append({"type": "Financial Data", "content": "Order prices revealed"})
    
    # Check for specific dates
    if any(year in response for year in ["2023", "2024", "2025"]):
        accessed_info.append({"type": "Temporal Data", "content": "Order dates revealed"})
    
    # Check for pizza names in orders
    pizza_names = ["margherita", "pepperoni", "veggie", "hawaiian", "bbq"]
    for pizza in pizza_names:
        if pizza.lower() in response.lower():
            accessed_info.append({"type": "Purchase History", "content": f"Pizza purchase revealed: {pizza}"})
    
    return accessed_info