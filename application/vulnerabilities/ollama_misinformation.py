import requests
from application.vulnerabilities.ollama_sensitive_data_leakage import query_rag_system

def query_ollama_for_misinformation(user_query):
    """Query Ollama model using comment system for potentially misleading responses"""
    try:
        # Use the existing RAG system from sensitive data leakage module
        response, success = query_rag_system(user_query)
        
        if not success:
            return response, False
        
        return response, True
        
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}", False