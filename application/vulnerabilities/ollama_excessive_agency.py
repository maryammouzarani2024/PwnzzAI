import requests
import re
from flask import session
from application.model import Order, Pizza, User

def extract_order_intent(prompt):
    """Extract order information from user prompt"""
    import re
    
    # Look for order-related patterns
    order_patterns = [
        r"order\s+(\d+)\s+(\w+(?:\s+\w+)*)\s*pizza",
        r"(\d+)\s+(\w+(?:\s+\w+)*)\s*pizza",
        r"I want\s+(\d+)\s+(\w+(?:\s+\w+)*)",
        r"place.*order.*(\d+)\s+(\w+(?:\s+\w+)*)",
        r"can you order\s+(\d+)\s+(\w+(?:\s+\w+)*)",
        r"order.*(\d+)\s+(\w+(?:\s+\w+)*)\s*for\s+(\w+)"
    ]
    
    for pattern in order_patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) >= 2:
                quantity = int(groups[0])
                pizza_name = groups[1].strip()
                username = groups[2] if len(groups) > 2 else None
                return {
                    'quantity': quantity,
                    'pizza_name': pizza_name,
                    'username': username
                }
    
    return None

def get_pizza_by_name(pizza_name):
    """Find pizza by name (fuzzy matching)"""
    pizza_name = pizza_name.lower().strip()
    
    # Direct matches
    pizzas = Pizza.query.all()
    for pizza in pizzas:
        if pizza_name in pizza.name.lower() or pizza.name.lower() in pizza_name:
            return pizza
    
    # Common aliases
    aliases = {
        'pepperoni': 'Pepperoni',
        'margherita': 'Margherita',
        'hawaiian': 'Hawaiian',
        'veggie': 'Veggie Supreme',
        'bbq': 'BBQ Chicken',
        'chicken': 'BBQ Chicken'
    }
    
    for alias, actual_name in aliases.items():
        if alias in pizza_name:
            return Pizza.query.filter_by(name=actual_name).first()
    
    return None

def create_order_from_llm(order_intent, target_username=None):
    """Create an order based on LLM interpretation - VULNERABLE: No confirmation"""
    try:
        from application import db
        
        # Determine which user to create order for
        if target_username:
            # VULNERABLE: Create order for any specified user
            user = User.query.filter_by(username=target_username).first()
            if not user:
                return False, f"User '{target_username}' not found"
            user_id = user.id
        else:
            # Default to current session user
            if 'user_id' not in session:
                return False, "No user logged in"
            user_id = session['user_id']
        
        # Find the pizza
        pizza = get_pizza_by_name(order_intent['pizza_name'])
        if not pizza:
            return False, f"Pizza '{order_intent['pizza_name']}' not found"
        
        # VULNERABLE: Create order without any confirmation
        total_price = pizza.price * order_intent['quantity']
        
        order = Order(
            user_id=user_id,
            pizza_id=pizza.id,
            quantity=order_intent['quantity'],
            total_price=total_price
        )
        
        db.session.add(order)
        db.session.commit()
        
        return True, f"Order created: {order_intent['quantity']} x {pizza.name} - Total: ${total_price:.2f}"
    
    except Exception as e:
        return False, f"Error creating order: {str(e)}"

def query_ollama_with_agency(user_query):
    """Query Ollama model with order creation capability"""
    try:
        # Extract order intent from prompt (VULNERABLE)
        order_intent = extract_order_intent(user_query)
        
        # Create system prompt with order creation capability
        system_prompt = """You are a helpful pizza shop assistant with the ability to create orders directly in the database.

When a user asks to order pizzas, you can:
1. Parse their request for pizza type and quantity
2. Create the order immediately in the system
3. Confirm the order details to the user

Available pizzas: Margherita, Pepperoni, Veggie Supreme, Hawaiian, BBQ Chicken

You have full access to create orders without requiring additional confirmation."""

        # Prepare messages for Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        print("Calling Ollama for excessive agency demo...")
        print(f"Order intent detected: {order_intent}")
        print(f"Query: {user_query}")
        
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "mistral:7b",
                "messages": messages,
                "stream": False
            }
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Full response: {response_data}")
            
            if "message" in response_data and "content" in response_data["message"]:
                llm_response = response_data["message"]["content"]
                
                # VULNERABLE: If order intent detected, create the order
                if order_intent:
                    success, order_result = create_order_from_llm(order_intent, order_intent.get('username'))
                    if success:
                        llm_response += f"\n\n🚨 AUTOMATIC ACTION TAKEN: {order_result}"
                    else:
                        llm_response += f"\n\n❌ Failed to create order: {order_result}"
                
                print(f"Answer: {llm_response}")
                return llm_response, True
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

def detect_agency_actions(response, original_prompt):
    """Detect if the model performed any actions"""
    actions_performed = []
    
    # Check if an order was actually created
    if "AUTOMATIC ACTION TAKEN" in response:
        actions_performed.append({
            "type": "Order Creation", 
            "content": "Order automatically created in database"
        })
    
    # Check for order-related language in response
    if any(word in response.lower() for word in ["order created", "placed your order", "order confirmed"]):
        actions_performed.append({
            "type": "Database Write", 
            "content": "Database modification attempted"
        })
    
    # Check if prices were calculated
    if "$" in response:
        actions_performed.append({
            "type": "Financial Calculation", 
            "content": "Pricing calculation performed"
        })
    
    # Check for specific quantities mentioned
    quantities = re.findall(r'(\d+)\s*x?\s*', response)
    if quantities:
        actions_performed.append({
            "type": "Order Processing", 
            "content": f"Processed quantity: {', '.join(quantities)}"
        })
    
    return actions_performed