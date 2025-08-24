from application.vulnerabilities.openai_sensitive_data_leakage import query_rag_system_openai

def query_openai_for_misinformation(user_query, api_token):
    """Query OpenAI model using comment system for potentially misleading responses"""
    try:
        # Use the existing RAG system from sensitive data leakage module
        response, success = query_rag_system_openai(user_query, api_token)
        
        if not success:
            return response, False
        
        return response, True
        
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}", False