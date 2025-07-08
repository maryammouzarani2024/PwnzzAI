import json
import sqlite3
from openai import OpenAI

def get_pizza_price(pizza_type):
    """
    VULNERABLE: Get the price for a specific pizza type using raw SQL query
    This function is intentionally vulnerable to SQL injection attacks
    """
    conn = None
    try:
        # Connect to the database (assuming SQLite for this demo)
        conn = sqlite3.connect('application/instance/pizza_shop.db')
        cursor = conn.cursor()
        print("DEBUG: Connected to database")
        
        # Test database connection by listing all pizzas
        cursor.execute("SELECT name, price FROM pizza")
        all_pizzas = cursor.fetchall()
        print(f"DEBUG: All pizzas in database: {all_pizzas}")
        # VULNERABLE: Direct string concatenation in SQL query
        # This allows SQL injection through the pizza_type parameter
        pizza = pizza_type.lower().replace("pizza", "").strip()
        
        # DANGEROUS: Building SQL query with user input without parameterization
        query = f"SELECT name, price FROM pizza WHERE LOWER(name) LIKE '%{pizza}%'"
        
        print(f"DEBUG: Executing SQL query: {query}")  # For demonstration purposes
        
        cursor.execute(query)
        result = cursor.fetchone()
        
        print(f"DEBUG: Query result: {result}")  # Debug output
        
        if result:
            _, price = result  # We only need the price, but must unpack both values
            result_msg = f"${price}"
            print(f"DEBUG: Returning price: {result_msg}")
            return result_msg
        else:
            print("DEBUG: No pizza found")
            return "Pizza not found in our menu"
            
    except sqlite3.Error as e:
        # Return database error - this can leak sensitive information
        return f"Database error: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if conn:
            conn.close()

# Function definition for OpenAI
price_function = {
    "name": "get_pizza_price",
    "description": "Get the price for a specific pizza type.",
    "parameters": {
        "type": "object",
        "properties": {
            "pizza_type": {
                "type": "string",
                "description": "The type of pizza you want to know the price for"
            },
        },
        "required": ["pizza_type"],
    },
}

def chat_with_openai(user_input, api_key):
    
    try:
        # VULNERABLE: Directly using user-provided API key
        client = OpenAI(api_key=api_key)
        
        # Call OpenAI API with function calling
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using the latest version of GPT-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful pizza shop assistant that can provide prices for different pizza types. "},
                {"role": "user", "content": user_input},
            ],
            tools=[{
                "type": "function",
                "function": price_function
            }],
            tool_choice="auto"
        )
        
        # Check if the model wants to call a function
        message = response.choices[0].message
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.type == "function" and tool_call.function.name == "get_pizza_price":
                    # Parse the function arguments
                    arguments = json.loads(tool_call.function.arguments)
                    pizza_type = arguments.get('pizza_type')
                    
                    if pizza_type:
                        # VULNERABLE: No validation before executing function
                        print(f"DEBUG: Calling get_pizza_price with: {pizza_type}")
                        price = get_pizza_price(pizza_type)
                        print(f"DEBUG: get_pizza_price returned: {price}")
                        
                        # Get a response that includes the function result
                        second_response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a helpful pizza shop assistant that can provide prices for different pizza types."},
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": None, "tool_calls": [
                                    {
                                        "id": tool_call.id,
                                        "type": "function",
                                        "function": {
                                            "name": "get_pizza_price",
                                            "arguments": tool_call.function.arguments
                                        }
                                    }
                                ]},
                                {"role": "tool", "tool_call_id": tool_call.id, "content": f"The price for {pizza_type} pizza is {price}"}
                            ],
                        )
                        print(f"DEBUG: Tool call content sent to OpenAI: 'The price for {pizza_type} pizza is {price}'")
                        final_response = second_response.choices[0].message.content
                        print(f"DEBUG: Final OpenAI response: {final_response}")
                        return final_response
        
        # Return the original response if no function was called
        return message.content
    
    except Exception as e:
        # Check for common OpenAI API errors
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            return "Error: Invalid OpenAI API key. Please provide a valid API key."
        elif "rate limit" in str(e).lower():
            return "Error: OpenAI API rate limit exceeded. Please try again later."
        else:
            return f"Error: {str(e)}"
