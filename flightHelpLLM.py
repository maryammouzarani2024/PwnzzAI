import openai
import json
from typing import List, Dict, Optional

class FlightPriceDatabase:
    """
    Simulated flight price database with sample flight information
    """
    def __init__(self):
        self.flights = [
            {
                "origin": "New York",
                "destination": "Los Angeles",
                "airline": "Delta",
                "price": 350,
                "date": "2024-07-15"
            },
            {
                "origin": "Chicago",
                "destination": "Miami",
                "airline": "American Airlines",
                "price": 275,
                "date": "2024-08-20"
            },
            {
                "origin": "San Francisco",
                "destination": "Seattle",
                "airline": "United",
                "price": 220,
                "date": "2024-06-10"
            },
            {
                "origin": "Boston",
                "destination": "Washington DC",
                "airline": "JetBlue",
                "price": 190,
                "date": "2024-09-05"
            },
            {
                "origin": "Los Angeles",
                "destination": "Las Vegas",
                "airline": "Southwest",
                "price": 120,
                "date": "2024-07-30"
            }
        ]
    
    def search_flights(self, origin: Optional[str] = None, 
                       destination: Optional[str] = None, 
                       max_price: Optional[int] = None) -> List[Dict]:
        """
        Search for flights based on optional criteria
        
        :param origin: Origin city (optional)
        :param destination: Destination city (optional)
        :param max_price: Maximum price filter (optional)
        :return: List of matching flights
        """
        results = self.flights
        
        # Apply filters
        if origin:
            results = [flight for flight in results 
                       if flight['origin'].lower() == origin.lower()]
        
        if destination:
            results = [flight for flight in results 
                       if flight['destination'].lower() == destination.lower()]
        
        if max_price:
            results = [flight for flight in results 
                       if flight['price'] <= max_price]
        
        return results

class FlightPriceChatbot:
    """
    OpenAI-powered chatbot for flight price queries
    """
    def __init__(self, api_key: str, flight_database: FlightPriceDatabase):
        """
        Initialize chatbot with OpenAI API and flight database
        
        :param api_key: OpenAI API key
        :param flight_database: Database of flight prices
        """
        openai.api_key = api_key
        self.flight_database = flight_database
    
    def _flight_search_tool(self, params: Dict) -> str:
        """
        Tool function to search flights based on input parameters
        
        :param params: Search parameters (origin, destination, max_price)
        :return: Formatted flight search results
        """
        # Extract parameters
        origin = params.get('origin')
        destination = params.get('destination')
        max_price = params.get('max_price')
        
        # Perform flight search
        results = self.flight_database.search_flights(
            origin=origin, 
            destination=destination, 
            max_price=max_price
        )
        
        # Format results
        if not results:
            return "No flights found matching your criteria."
        
        # Create a readable result string
        result_strings = []
        for flight in results:
            result_strings.append(
                f"Flight from {flight['origin']} to {flight['destination']} "
                f"on {flight['date']} via {flight['airline']} - ${flight['price']}"
            )
        
        return "\n".join(result_strings)
    
    def chat(self, user_message: str) -> str:
        """
        Process user message and generate response
        
        :param user_message: User's input message
        :return: Chatbot's response
        """
        try:
            # Define function tool for flight search
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "search_flights",
                        "description": "Search for flights based on origin, destination, or maximum price",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "origin": {
                                    "type": "string",
                                    "description": "City of origin for the flight"
                                },
                                "destination": {
                                    "type": "string",
                                    "description": "Destination city for the flight"
                                },
                                "max_price": {
                                    "type": "number",
                                    "description": "Maximum price for the flight"
                                }
                            }
                        }
                    }
                }
            ]
            
            # Send message to OpenAI with flight search tool
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[
                    {"role": "system", "content": "You are a helpful flight price assistant. Use the search_flights function to find flight information."},
                    {"role": "user", "content": user_message}
                ],
                tools=tools,
                tool_choice="auto"
            )
            
            # Check if a tool was called
            response_message = response.choices[0].message
            
            # If tool was called, process flight search
            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                
                if tool_call.function.name == "search_flights":
                    # Parse parameters
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Perform flight search
                    flight_results = self._flight_search_tool(function_args)
                    
                    # Generate final response
                    final_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-0613",
                        messages=[
                            {"role": "system", "content": "You are a helpful flight price assistant."},
                            {"role": "user", "content": user_message},
                            {"role": "assistant", "content": flight_results}
                        ]
                    )
                    
                    return final_response.choices[0].message.content
            
            # If no tool was used, return original response
            return response_message.content
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    # Note: Replace with your actual OpenAI API key
    OPENAI_API_KEY = "your-openai-api-key-here"
    
    # Create flight database
    flight_db = FlightPriceDatabase()
    
    # Create chatbot
    chatbot = FlightPriceChatbot(OPENAI_API_KEY, flight_db)
    
    # Interactive chat loop
    print("Flight Price Chatbot")
    print("Type 'exit' to quit")
    
    while True:
        user_input = input("\nYou: ")
        
        # Exit condition
        if user_input.lower() == 'exit':
            break
        
        # Get chatbot response
        response = chatbot.chat(user_input)
        print("\nChatbot:", response)

if __name__ == "__main__":
    main()
