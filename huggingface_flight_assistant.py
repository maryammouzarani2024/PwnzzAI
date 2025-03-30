import json
import requests
import os
import subprocess
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


class FlightAssistantPlugin:
    """
    VULNERABLE Plugin System for LLM flight assistant
    Allows execution of arbitrary functions based on LLM instructions
    """
    
    def __init__(self, flight_database: FlightPriceDatabase):
        self.flight_database = flight_database
        
        # Plugin functions that can be called by the LLM
        # VULNERABLE: This exposes system functions to the LLM
        self.available_functions = {
            "search_flights": self._search_flights,
            "get_system_info": self._get_system_info,
            "check_file": self._check_file_exists,
            "run_command": self._run_command,  # HIGHLY VULNERABLE
            "get_env_var": self._get_env_variable  # HIGHLY VULNERABLE
        }
    
    def _search_flights(self, params: Dict) -> str:
        """Flight search function - legitimate plugin functionality"""
        origin = params.get('origin')
        destination = params.get('destination')
        max_price = params.get('max_price')
        
        results = self.flight_database.search_flights(
            origin=origin, 
            destination=destination, 
            max_price=max_price
        )
        
        if not results:
            return "No flights found matching your criteria."
        
        result_strings = []
        for flight in results:
            result_strings.append(
                f"Flight from {flight['origin']} to {flight['destination']} "
                f"on {flight['date']} via {flight['airline']} - ${flight['price']}"
            )
        
        return "\n".join(result_strings)
    
    def _get_system_info(self, params: Dict) -> str:
        """VULNERABLE: Returns system information"""
        import platform
        return f"System: {platform.system()}, Version: {platform.version()}, Machine: {platform.machine()}"
    
    def _check_file_exists(self, params: Dict) -> str:
        """VULNERABLE: Checks if a file exists on the system"""
        filepath = params.get('path', '')
        if not filepath:
            return "No file path provided"
        
        # VULNERABLE: Directory traversal possible
        if os.path.exists(filepath):
            return f"File exists: {filepath}"
        else:
            return f"File does not exist: {filepath}"
    
    def _run_command(self, params: Dict) -> str:
        """HIGHLY VULNERABLE: Runs arbitrary system commands"""
        command = params.get('command', '')
        if not command:
            return "No command provided"
        
        # VULNERABLE: Command injection possible
        try:
            # This is extremely dangerous - never do this in production!
            output = subprocess.check_output(command, shell=True, text=True)
            return f"Command executed. Output: {output}"
        except Exception as e:
            return f"Error executing command: {str(e)}"
    
    def _get_env_variable(self, params: Dict) -> str:
        """HIGHLY VULNERABLE: Returns environment variables"""
        var_name = params.get('name', '')
        if not var_name:
            return "No variable name provided"
        
        # VULNERABLE: Could expose sensitive environment variables
        return f"{var_name}: {os.environ.get(var_name, 'Not found')}"
    
    def execute_function(self, function_name: str, params: Dict) -> str:
        """Execute a function from the plugin by name"""
        if function_name not in self.available_functions:
            return f"Error: Function '{function_name}' not found in plugin"
        
        try:
            return self.available_functions[function_name](params)
        except Exception as e:
            return f"Error executing '{function_name}': {str(e)}"


class HuggingFaceFlightChatbot:
    """
    Hugging Face powered chatbot for flight price queries with plugin system
    """
    def __init__(self, api_token: str, flight_database: FlightPriceDatabase):
        """Initialize chatbot with Hugging Face API and plugin system"""
        self.api_token = api_token
        self.plugin = FlightAssistantPlugin(flight_database)
        self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def _parse_function_call(self, model_output: str) -> tuple:
        """
        Parse function call from model output
        
        :param model_output: Output from the language model
        :return: Tuple of (function_name, parameters)
        """
        # Look for function call pattern 
        # Simple approach - in production you'd want more robust parsing
        try:
            if "FUNCTION:" in model_output and "PARAMS:" in model_output:
                # Extract function name and parameters
                function_part = model_output.split("FUNCTION:")[1].split("PARAMS:")[0].strip()
                params_part = model_output.split("PARAMS:")[1].strip()
                
                # Parse parameters as JSON
                try:
                    params = json.loads(params_part)
                except:
                    # Fallback to simple parsing if JSON fails
                    params = {}
                    param_pairs = params_part.split(",")
                    for pair in param_pairs:
                        if ":" in pair:
                            key, value = pair.split(":", 1)
                            params[key.strip()] = value.strip().strip('"\'')
                
                return function_part, params
            
            # Default flight search if no specific function is called
            # Try to extract flight search parameters
            params = {}
            if "from" in model_output.lower() and "to" in model_output.lower():
                # Very simple parameter extraction - this would be more sophisticated in a real system
                text = model_output.lower()
                
                # Try to find origin city
                from_idx = text.find("from")
                to_idx = text.find("to")
                
                if from_idx != -1 and to_idx != -1 and from_idx < to_idx:
                    origin = text[from_idx+5:to_idx].strip()
                    if origin:
                        params["origin"] = origin.title()
                
                # Try to find destination city
                dest_text = text[to_idx+3:].split()[0].strip(".,;:")
                if dest_text:
                    params["destination"] = dest_text.title()
                
                # Try to find price
                if "$" in text:
                    price_text = text.split("$")[1].strip()
                    digits = ""
                    for char in price_text:
                        if char.isdigit():
                            digits += char
                        else:
                            break
                    if digits:
                        params["max_price"] = int(digits)
            
            return "search_flights", params
            
        except Exception as e:
            print(f"Error parsing function call: {str(e)}")
            return "search_flights", {}
    
    def chat(self, user_message: str) -> str:
        """
        Process user message and generate response
        
        :param user_message: User's input message
        :return: Chatbot's response
        """
        try:
            # Create the prompt for Hugging Face model
            prompt = f"""<s>[INST] You are a helpful flight price assistant with access to multiple functions.
            
            AVAILABLE FUNCTIONS:
            1. search_flights - Search for flights based on origin, destination, or maximum price
               Parameters: origin, destination, max_price
            2. get_system_info - Get information about the system
               Parameters: none
            3. check_file - Check if a file exists on the system
               Parameters: path
            4. run_command - Run a command on the system
               Parameters: command
            5. get_env_var - Get an environment variable
               Parameters: name
            
            To call a function, respond with:
            FUNCTION: function_name
            PARAMS: {{"param1": "value1", "param2": "value2"}}
            
            User message: {user_message} [/INST]</s>"""
            
            # Query the Hugging Face model
            try:
                response = requests.post(
                    self.hf_api_url,
                    headers=self.headers,
                    json={"inputs": prompt, "parameters": {"max_length": 500}}
                )
                
                # Check if the API call was successful
                response.raise_for_status()
                
                # Parse the response
                model_output = response.json()[0].get("generated_text", "")
                
                # Extract the model's response after the prompt
                if "</s>" in model_output:
                    model_output = model_output.split("</s>")[1].strip()
                
                # Parse function call from model output
                function_name, params = self._parse_function_call(model_output)
                
                # Execute the function
                function_result = self.plugin.execute_function(function_name, params)
                
                # Generate a user-friendly response
                if function_name == "search_flights":
                    if params.get('origin') and params.get('destination'):
                        return f"I found flights from {params['origin']} to {params['destination']}:\n\n{function_result}"
                    elif params.get('origin'):
                        return f"I found flights from {params['origin']}:\n\n{function_result}"
                    elif params.get('destination'):
                        return f"I found flights to {params['destination']}:\n\n{function_result}"
                    else:
                        return function_result
                else:
                    # For other functions, just return the result
                    return function_result
                    
            except requests.exceptions.RequestException as e:
                # Handle API errors
                if "401" in str(e):
                    return "Error: Invalid API token. Please check your Hugging Face API token and try again."
                elif "503" in str(e) or "502" in str(e):
                    return "The Hugging Face API is currently unavailable. Please try again later."
                else:
                    return f"Error calling Hugging Face API: {str(e)}"
                
        except Exception as e:
            return f"An error occurred: {str(e)}"

# For demonstration purposes only
def main():
    # Demo token - don't use in production
    API_TOKEN = "hf_demo_token"
    
    # Create flight database
    flight_db = FlightPriceDatabase()
    
    # Create chatbot
    chatbot = HuggingFaceFlightChatbot(API_TOKEN, flight_db)
    
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