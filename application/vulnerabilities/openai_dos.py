"""
Simple OpenAI Chatbot Implementation
This module provides a simple chat interface using OpenAI's API.
"""

from application.llm_chat import chat_completion
from application.provider_config import lab_cloud_llm_model_default


def chat_with_openai(user_message: str, api_key: str) -> str:
    """
    Simple chat function that sends a message to OpenAI API.
    
    Args:
        user_message: The user's message
        api_key: OpenAI API key
        
    Returns:
        The AI's response
    """
    return chat_completion(
        [
            {"role": "system", "content": "You are a helpful pizza assistant. Help users with pizza recipes and information. Keep responses concise and helpful."},
            {"role": "user", "content": user_message}
        ],
        api_key=api_key,
        model=lab_cloud_llm_model_default(),
        max_tokens=500,
        temperature=0.7,
    )