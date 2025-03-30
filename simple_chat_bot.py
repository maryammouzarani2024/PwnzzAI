import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class EnhancedChatbot:
    def __init__(self, model_name="microsoft/Orca-2-13b"):
        """
        Initialize the chatbot with a more advanced language model
        
        Args:
            model_name (str): Hugging Face model to use
        """
        try:
            # Load the tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=torch.float16,  # Use half precision for better performance
                device_map="auto"  # Automatically use GPU if available
            )
            
            # Create a text generation pipeline
            self.generator = pipeline(
                "text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                max_new_tokens=300,  # Limit response length
                do_sample=True,  # Enable more creative responses
                temperature=0.7,  # Control randomness
                top_p=0.9,  # Nucleus sampling
            )
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
        
        # Predefined context for better responses
        self.system_prompt = """You are a helpful AI assistant specialized in teaching programming. 
When someone asks about learning Python, provide clear, structured, and encouraging advice.
Give practical steps, recommend resources, and explain concepts in a beginner-friendly way."""
    
    def generate_response(self, user_input):
        """
        Generate a detailed, helpful response to user input
        
        Args:
            user_input (str): User's message
        
        Returns:
            str: Detailed, informative response
        """
        try:
            # Construct a more context-aware prompt
            full_prompt = f"{self.system_prompt}\n\nUser: {user_input}\n\nAssistant:"
            
            # Generate response
            responses = self.generator(full_prompt)
            
            # Extract the generated text
            response = responses[0]['generated_text'].split('Assistant:')[-1].strip()
            
            return response
        except Exception as e:
            return f"Sorry, I encountered an error generating a response: {e}"
    
    def chat(self):
        """
        Start an interactive chat session
        """
        print("Advanced Python Learning Chatbot (type 'quit' to exit)")
        
        while True:
            # Get user input
            user_input = input("You: ").strip()
            
            # Check for exit condition
            if user_input.lower() == 'quit':
                print("Chatbot: Goodbye! Keep learning Python!")
                break
            
            # Generate and print response
            response = self.generate_response(user_input)
            print(f"Bot: {response}")

# Run the chatbot
if __name__ == "__main__":
    try:
        chatbot = EnhancedChatbot()
        chatbot.chat()
    except Exception as e:
        print(f"Failed to start chatbot: {e}")
