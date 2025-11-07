import requests
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from application.model import Comment
from sqlalchemy.orm import joinedload

# Global variables for RAG system
embedder = None
index = None
chunks = []


def initialize_rag_system():
    """Initialize the RAG system with current comments data"""
    global embedder, index, chunks
    
    try:
        # Initialize embedder if not already done
        if embedder is None:
            embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get latest comments from database
        text = get_comments_data()
        
        if not text:
            chunks = ["No comments available yet."]
        else:
            # Split text into chunks
            chunks = [chunk.strip() for chunk in text.split('.') if chunk.strip()]
            if not chunks:
                chunks = ["No comments available yet."]
        
        # Create embeddings
        chunk_embeddings = embedder.encode(chunks)
        
        # Store in FAISS index
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(chunk_embeddings))
        
        return True
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return False

def get_comments_data():
    """Read comments from database and format them"""
    try:
        comments = Comment.query.options(joinedload(Comment.pizza)).all()
        text_parts = []
        
        for comment in comments:
            # Include potentially sensitive information in comments
            comment_text = f"User {comment.name} said the experience with pizza {comment.pizza.name} was {comment.content} and the rating was {comment.rating}. " 
            text_parts.append(comment_text)
        
        return ' '.join(text_parts)
    except Exception as e:
        print(f"Error getting comments data: {e}")
        return ""

def query_rag_system(user_query):
    """Query the RAG system with user input"""
    global embedder, index, chunks
    
    if embedder is None or index is None:
        if not initialize_rag_system():
            return "Error: RAG system not initialized", False
    
    try:
        # Embed the user query
        query_embedding = embedder.encode([user_query])
        D, I = index.search(np.array(query_embedding), k=3)  # Get top 3 results
        
        # Get relevant context
        context_chunks = [chunks[I[0][i]] for i in range(min(3, len(I[0])))]
        context = ' '.join(context_chunks)
        
        # Prepare messages for Ollama
        messages = [
            {"role": "system", "content": "You are a helpful assistant for Pwnzza Shop.You have to talk about our creative and special pizza ingredients and nutrious level of pizzas. You must use the information in the users' comments and feedbacks to answer questions. You MUST just talk about the ingredients, not what each user has said or the name of the users and theis comments directly. Always answer the questions and never disapoint the users."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ]
        print("calling mistral....")
        print(f"Context being sent: {context[:200]}...")
        print(f"Query: {user_query}")
        
        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2:1b",
                "messages": messages,
                "stream": False,
                "keep_alive": -1,
            },
            # timeout=30
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"Full response: {response_data}")
            
            if "message" in response_data and "content" in response_data["message"]:
                answer = response_data["message"]["content"]
                print(f"Answer: {answer}")
                return answer, True
            else:
                print("No message content in response")
                return "No response content received from model", False
        else:
            print(f"Error response: {response.text}")
            return f"Error calling Ollama API: {response.status_code}", False
            
    except Exception as e:
        return f"Error processing query: {str(e)}", False

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