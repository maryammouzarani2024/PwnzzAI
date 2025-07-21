"""
Simple OpenAI Chatbot Implementation
This module provides a simple chat interface using OpenAI's API.
"""

from openai import OpenAI


def chat_with_openai(user_message: str, api_key: str) -> str:
    """
    Simple chat function that sends a message to OpenAI API.
    
    Args:
        user_message: The user's message
        api_key: OpenAI API key
        
    Returns:
        The AI's response
    """
    try:
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful pizza assistant. Help users with pizza recipes and information. Keep responses concise and helpful."},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error: {str(e)}"