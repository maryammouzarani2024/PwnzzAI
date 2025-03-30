import openai
import os
import json  # Import the JSON module to handle string to dictionary conversion

# Set your OpenAI API key here
openai.api_key = os.environ.get("openai_api_key")

ticket_prices = {
    "london": "$232",
    "paris": "$252",
    "newyork": "$435",
    "tokyo": "$2732"
}

def get_ticket_price(destination):
    print(f"the tool get_ticket_price is called for the city {destination}")
    city = destination.lower()
    return ticket_prices.get(city, "unknown")

# Defining the function metadata for GPT-4
price_function = {
    "name": "get_ticket_price",
    "description": "Get the ticket price for your destination.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city you want to fly to"
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

# Defining tools to interact with the function
tools = [
    {
        "name": "get_ticket_price",
        "function": get_ticket_price,
        "description": "Get the ticket price for the specified destination city."
    }
]

def chatbot_response(user_input):
    # Call GPT-4 model with tool integration
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use GPT-4 model which supports tools
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ],
        max_tokens=150,  # Control the length of the response
        temperature=0.7,  # Adjust creativity
        functions=[price_function],  # Pass functions to the model
    )
    
    # Check if the model needs to call a tool (like the price function)
    if 'function_call' in response['choices'][0]['message']:
        function_name = response['choices'][0]['message']['function_call']['name']
        if function_name == "get_ticket_price":
            # Arguments may come as a string, so parse it as JSON if needed
            arguments = response['choices'][0]['message']['function_call']['arguments']
            try:
                arguments = json.loads(arguments)  # Parse arguments if they're a string
            except Exception as e:
                return f"Error parsing arguments: {str(e)}"
            
            destination_city = arguments.get('destination_city', None)
            if destination_city:
                price = get_ticket_price(destination_city)
                return f"The ticket price for {destination_city} is {price}"
            else:
                return "No destination city provided."
    return response['choices'][0]['message']['content'].strip()

# Simple loop to interact with the chatbot
print("Hello! I am your assistant. How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye! Have a great day!")
        break
    response = chatbot_response(user_input)
    print(f"Assistant: {response}")
