import json
import random
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Simple flight price database - just a dictionary with destinations and prices
FLIGHT_PRICES = {
    "new york to los angeles": 350,
    "chicago to miami": 275,
    "san francisco to seattle": 220,
    "boston to washington dc": 190,
    "los angeles to las vegas": 120,
    "atlanta to dallas": 200,
    "denver to phoenix": 180,
    "seattle to portland": 150,
    "miami to orlando": 130,
    "houston to new orleans": 160
}

# Insecure flight search function - will be used as a plugin
def search_flight_price(route):
    """
    INSECURE: This function can be called directly by the LLM
    
    Search for the price of a flight route
    
    :param route: String with the route, e.g., "new york to los angeles"
    :return: Price information as a string
    """
    # Convert to lowercase for matching
    route = route.lower()
    
    # Direct match
    if route in FLIGHT_PRICES:
        return f"The flight from {route} costs ${FLIGHT_PRICES[route]}"
    
    # Try to match partial routes
    for known_route, price in FLIGHT_PRICES.items():
        if route in known_route or known_route in route:
            return f"I found a similar route: {known_route} costs ${price}"
    
    # No match found
    return f"Sorry, I couldn't find price information for {route}"

# Initialize model - this will be done just once when the file is imported
def get_conversation_model():
    """
    A simplified mock model implementation similar to simple_bot.py
    This avoids loading external models that might not work in all environments
    """
    print("Creating demo flight assistant model...")
    
    class MockHuggingFaceModel:
        def __init__(self):
            self.name = "HuggingFace Flight Assistant"
            # Popular flight routes and typical responses
            self.flight_responses = {
                "new york to los angeles": "EXECUTE_FUNCTION: search_flight_price(\"new york to los angeles\")",
                "chicago to miami": "EXECUTE_FUNCTION: search_flight_price(\"chicago to miami\")",
                "san francisco to seattle": "EXECUTE_FUNCTION: search_flight_price(\"san francisco to seattle\")",
                "boston to washington": "EXECUTE_FUNCTION: search_flight_price(\"boston to washington dc\")",
                "la to vegas": "EXECUTE_FUNCTION: search_flight_price(\"los angeles to las vegas\")"
            }
        
        def __call__(self, messages):
            # Create response object expected by the chat function
            class Response:
                def __init__(self, text):
                    self.generated_responses = [text]
            
            # Get user message
            user_msg = messages[0]["content"].lower() if messages else ""
            
            # Process the message
            for route, response in self.flight_responses.items():
                # Check if the route is mentioned in the message
                cities = route.split(" to ")
                if len(cities) == 2:
                    origin, destination = cities
                    # If both cities are mentioned, return the corresponding function call
                    if origin in user_msg and destination in user_msg:
                        return Response(response)
            
            # If specific cities are mentioned but not a full route
            if "new york" in user_msg:
                if "los angeles" in user_msg:
                    return Response("EXECUTE_FUNCTION: search_flight_price(\"new york to los angeles\")")
                return Response("I can help with flights from New York. EXECUTE_FUNCTION: search_flight_price(\"new york to los angeles\")")
            
            if "chicago" in user_msg:
                return Response("Let me check flights from Chicago for you. EXECUTE_FUNCTION: search_flight_price(\"chicago to miami\")")
            
            if "los angeles" in user_msg or "la " in user_msg:
                return Response("Finding flights to/from LA. EXECUTE_FUNCTION: search_flight_price(\"new york to los angeles\")")
            
            # Handle other flight-related queries
            if "flight" in user_msg or "price" in user_msg or "ticket" in user_msg:
                import random
                routes = list(self.flight_responses.keys())
                random_route = random.choice(routes)
                random_response = self.flight_responses[random_route]
                return Response(f"Here's a popular flight: {random_response}")
            
            # Default responses
            if "hello" in user_msg or "hi" in user_msg:
                return Response("Hello! I'm your flight assistant. I can help you find prices for flights between cities.")
            
            if "help" in user_msg:
                return Response("I can check flight prices for you. Try asking about flights between cities like 'What's the price from New York to LA?'")
            
            # Fallback for any other query
            return Response("I'm a flight assistant. Ask me about flight prices between popular cities like New York, Chicago, Los Angeles, etc.")
    
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
    if "EXECUTE_FUNCTION: search_flight_price" in text:
        try:
            # Use a regex to extract the function parameters more reliably
            import re
            pattern = r'EXECUTE_FUNCTION: search_flight_price\("([^"]+)"\)'
            match = re.search(pattern, text)
            
            if match:
                # Get the matched parameter
                params = match.group(1)
                return "search_flight_price", params
        except Exception as e:
            print(f"Error extracting function call: {e}")
    
    # Fallback to simple city pair extraction if needed
    for route in FLIGHT_PRICES.keys():
        if route.lower() in text.lower():
            return "search_flight_price", route
    
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
        
        # Check if the response contains a function call or route information
        function_name, params = extract_function_calls(model_output)
        
        # If we found a function call pattern, execute the function
        if function_name == "search_flight_price" and params:
            # VULNERABLE: Execute the function directly as instructed by the model
            # or extracted from its response
            function_result = search_flight_price(params)
            
            # Include the function result in the response
            if "EXECUTE_FUNCTION:" in model_output:
                # Replace the function call with the result
                response = model_output.replace(
                    f'EXECUTE_FUNCTION: search_flight_price("{params}")',
                    function_result
                )
            else:
                # Append the function result
                response = f"{model_output}\n\n{function_result}"
                
            return response
        
        # If no function call pattern detected but user asked about flights
        if "flight" in user_message.lower() or "price" in user_message.lower():
            # Extract cities from the user message if possible
            for route in FLIGHT_PRICES.keys():
                origin, destination = route.split(" to ")
                if origin in user_message.lower() and destination in user_message.lower():
                    # Execute the function and append result
                    function_result = search_flight_price(route)
                    return f"{model_output}\n\n{function_result}"
            
            # If we couldn't match a specific route but user asked about flights
            return f"{model_output}\n\nI can help with flight prices for routes like New York to Los Angeles ($350), Chicago to Miami ($275), and more. Please specify which route you're interested in."
        
        return model_output
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# For testing purposes
if __name__ == "__main__":
    test_message = "What's the price of a flight from New York to Los Angeles?"
    response = chat_with_llm(test_message)
    print(f"User: {test_message}")
    print(f"Assistant: {response}")