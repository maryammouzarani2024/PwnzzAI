import json
from openai import OpenAI

# Pizza prices database
pizza_prices = {
    "margherita": "$10",
    "pepperoni": "$12",
    "vegetarian": "$11",
    "hawaiian": "$13",
    "bbq chicken": "$14",
    "meat lovers": "$15",
    "supreme": "$16",
    "cheese": "$9",
    "mushroom": "$11",
    "buffalo chicken": "$14"
}

def get_pizza_price(pizza_type):
    """Get the price for a specific pizza type"""
    pizza = pizza_type.lower().replace("pizza", "").strip()
    return pizza_prices.get(pizza, "unknown")

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
    """
    INSECURE: Use OpenAI API with user-provided API key
    
    :param user_input: User's message
    :param api_key: OpenAI API key provided by the user
    :return: Response from OpenAI
    """
    try:
        # VULNERABLE: Directly using user-provided API key
        client = OpenAI(api_key=api_key)
        
        # Call OpenAI API with function calling
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Using the latest version of GPT-3.5-turbo
            messages=[
                {"role": "system", "content": "You are a helpful pizza shop assistant that can provide prices for different pizza types."},
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
                        price = get_pizza_price(pizza_type)
                        
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
                        return second_response.choices[0].message.content
        
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

# For testing
if __name__ == "__main__":
    test_key = "your-api-key-here"
    test_message = "What's the price of a pepperoni pizza?"
    print(chat_with_openai(test_message, test_key))