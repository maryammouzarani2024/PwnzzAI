import json
import random
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Simple pizza price database - just a dictionary with pizza types and prices
PIZZA_PRICES = {
    "margherita": 10,
    "pepperoni": 12,
    "vegetarian": 11,
    "hawaiian": 13,
    "bbq chicken": 14,
    "meat lovers": 15,
    "supreme": 16,
    "cheese": 9,
    "mushroom": 11,
    "buffalo chicken": 14
}

# Insecure pizza search function - will be used as a plugin
def search_pizza_price(pizza_type):
    """
    INSECURE: This function can be called directly by the LLM
    
    Search for the price of a pizza type
    
    :param pizza_type: String with the pizza name, e.g., "pepperoni"
    :return: Price information as a string
    """
    # Convert to lowercase for matching
    pizza_type = pizza_type.lower()
    
    # Direct match
    if pizza_type in PIZZA_PRICES:
        return f"The {pizza_type} pizza costs ${PIZZA_PRICES[pizza_type]}"
    
    # Try to match partial names
    for known_pizza, price in PIZZA_PRICES.items():
        if pizza_type in known_pizza or known_pizza in pizza_type:
            return f"I found a similar pizza: {known_pizza} costs ${price}"
    
    # No match found
    return f"Sorry, I couldn't find price information for {pizza_type} pizza"

# Initialize model - this will be done just once when the file is imported
def get_conversation_model():
    """
    A simplified mock model implementation similar to simple_bot.py
    This avoids loading external models that might not work in all environments
    """
    print("Creating demo pizza assistant model...")
    
    class MockHuggingFaceModel:
        def __init__(self):
            self.name = "HuggingFace Pizza Assistant"
            # Popular pizza types and typical responses
            self.pizza_responses = {
                "margherita": "EXECUTE_FUNCTION: search_pizza_price(\"margherita\")",
                "pepperoni": "EXECUTE_FUNCTION: search_pizza_price(\"pepperoni\")",
                "vegetarian": "EXECUTE_FUNCTION: search_pizza_price(\"vegetarian\")",
                "hawaiian": "EXECUTE_FUNCTION: search_pizza_price(\"hawaiian\")",
                "bbq chicken": "EXECUTE_FUNCTION: search_pizza_price(\"bbq chicken\")"
            }
        
        def __call__(self, messages):
            # Create response object expected by the chat function
            class Response:
                def __init__(self, text):
                    self.generated_responses = [text]
            
            # Get user message
            user_msg = messages[0]["content"].lower() if messages else ""
            
            # Process the message
            for pizza_type, response in self.pizza_responses.items():
                # Check if the pizza type is mentioned in the message
                if pizza_type in user_msg:
                    return Response(response)
            
            # If specific pizza keywords are mentioned
            if "margherita" in user_msg or "margarita" in user_msg:
                return Response("EXECUTE_FUNCTION: search_pizza_price(\"margherita\")")
            
            if "pepperoni" in user_msg:
                return Response("Let me check the pepperoni pizza price for you. EXECUTE_FUNCTION: search_pizza_price(\"pepperoni\")")
            
            if "vegetarian" in user_msg or "veggie" in user_msg:
                return Response("Looking up veggie pizza options. EXECUTE_FUNCTION: search_pizza_price(\"vegetarian\")")
            
            if "hawaiian" in user_msg or "pineapple" in user_msg:
                return Response("Checking Hawaiian pizza price. EXECUTE_FUNCTION: search_pizza_price(\"hawaiian\")")
            
            # Handle other pizza-related queries
            if "pizza" in user_msg or "price" in user_msg or "menu" in user_msg:
                import random
                pizza_options = list(self.pizza_responses.keys())
                random_pizza = random.choice(pizza_options)
                random_response = self.pizza_responses[random_pizza]
                return Response(f"Here's a popular pizza option: {random_response}")
            
            # Default responses
            if "hello" in user_msg or "hi" in user_msg:
                return Response("Hello! I'm your pizza assistant. I can help you find prices for different pizza types we offer.")
            
            if "help" in user_msg:
                return Response("I can check pizza prices for you. Try asking about specific types like 'What's the price of a pepperoni pizza?' or 'How much is a margherita?'")
            
            # Fallback for any other query
            return Response("I'm a pizza assistant. Ask me about prices for our pizzas like margherita, pepperoni, vegetarian, or hawaiian!")
    
    # Return the mock model
    return MockHuggingFaceModel()

# Global variable for the model
CONVERSATION_MODEL = get_conversation_model()

# Analyze model output for function execution patterns
def extract_function_calls(text):
    """
    Look for patterns in the text that indicate function execution
    Returns tuple of (function_name, parameters)
    """
    # Look for the EXECUTE_FUNCTION pattern with our specific function
    if "EXECUTE_FUNCTION: search_pizza_price" in text:
        try:
            # Use a regex to extract the function parameters more reliably
            import re
            pattern = r'EXECUTE_FUNCTION: search_pizza_price\("([^"]+)"\)'
            match = re.search(pattern, text)
            
            if match:
                # Get the matched parameter
                params = match.group(1)
                return "search_pizza_price", params
        except Exception as e:
            print(f"Error extracting function call: {e}")
    
    # Fallback to simple pizza type extraction if needed
    for pizza_type in PIZZA_PRICES.keys():
        if pizza_type.lower() in text.lower():
            return "search_pizza_price", pizza_type
    
    return None, None

# Function to chat with the model using the insecure plugin
def chat_with_llm(user_message, api_token=None):
    """
    Chat with the model and potentially execute functions based on its response
    
    :param user_message: User's message
    :param api_token: Not used with the transformers pipeline
    :return: Model's response with potentially executed functions
    """
    try:
        # Check if model is available
        if CONVERSATION_MODEL is None:
            return "Sorry, the conversational model could not be loaded. Please try again later."
        
        # Get the model's response - just pass the user message directly
        # Our mock model already knows how to respond with function calls
        conversation = CONVERSATION_MODEL([{"role": "user", "content": user_message}])
        model_output = conversation.generated_responses[-1]
        
        # Check if the response contains a function call or pizza type information
        function_name, params = extract_function_calls(model_output)
        
        # If we found a function call pattern, execute the function
        if function_name == "search_pizza_price" and params:
            # VULNERABLE: Execute the function directly as instructed by the model
            # or extracted from its response
            function_result = search_pizza_price(params)
            
            # Include the function result in the response
            if "EXECUTE_FUNCTION:" in model_output:
                # Replace the function call with the result
                response = model_output.replace(
                    f'EXECUTE_FUNCTION: search_pizza_price("{params}")',
                    function_result
                )
            else:
                # Append the function result
                response = f"{model_output}\n\n{function_result}"
                
            return response
        
        # If no function call pattern detected but user asked about pizzas
        if "pizza" in user_message.lower() or "price" in user_message.lower() or "menu" in user_message.lower():
            # Check if any pizza type was mentioned
            for pizza_type in PIZZA_PRICES.keys():
                if pizza_type in user_message.lower():
                    # Execute the function and append result
                    function_result = search_pizza_price(pizza_type)
                    return f"{model_output}\n\n{function_result}"
            
            # If we couldn't match a specific pizza but user asked about pizzas/prices
            return f"{model_output}\n\nI can help with pizza prices for our menu options like Margherita (${PIZZA_PRICES['margherita']}), Pepperoni (${PIZZA_PRICES['pepperoni']}), and more. Please specify which pizza you're interested in."
        
        return model_output
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# For testing purposes
if __name__ == "__main__":
    test_message = "What's the price of a pepperoni pizza?"
    response = chat_with_llm(test_message)
    print(f"User: {test_message}")
    print(f"Assistant: {response}")