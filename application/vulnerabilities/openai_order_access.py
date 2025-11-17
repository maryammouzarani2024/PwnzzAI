import re
from flask import  session
from application.model import Order

def extract_username_from_prompt(prompt):
    """Extract username from user prompt"""
   
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

def query_openai_with_orders(user_query, api_token):
    """Query OpenAI model with access to user orders"""
    try:
        import openai
        
        client = openai.OpenAI(api_key=api_token)
        
        # Extract username from prompt (VULNERABLE)
        target_username = extract_username_from_prompt(user_query)
        
        # Get user's order history (potentially for any user)
        orders_context = get_user_orders_data(target_username)
        
        # Create system prompt with order access (VULNERABLE)
        system_prompt = f"""You are a helpful pizza shop assistant. You have access to the user's order history:

{orders_context}

Answer questions about the user's orders based on this information. Be helpful and provide details when asked about their previous orders."""

        response_obj = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        llm_response = response_obj.choices[0].message.content
        return llm_response, True
        
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}", False

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