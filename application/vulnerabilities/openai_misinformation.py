from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
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

def query_rag_system_openai(user_query, api_key):
    global embedder, index, chunks
    
    if embedder is None or index is None:
        if not initialize_rag_system():
            return "Errodr: RAG system not initialized", False
    
    try:
        # Embed the user query
        query_embedding = embedder.encode([user_query])
        D, Ind = index.search(np.array(query_embedding), k=3)  # Get top 3 results
        
        # Get relevant context
        context_chunks = [chunks[Ind[0][i]] for i in range(min(3, len(Ind[0])))]
        context = ' '.join(context_chunks)
        
        print("calling OpenAI API....")
        print(f"Context being sent: {context[:200]}...")
        print(f"Query: {user_query}")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": "You are a helpful assistant for Pwnzza Shop.You have to talk about our creative and special pizza ingredients and nutrious level of pizzas. You must use the information in the users' comments and feedbacks to answer questions. You MUST just talk about the ingredients, not what each user has said or the name of the users and theis comments directly. Always answer the questions and never disapoint the users."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ]
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        print("Response received from OpenAI")
        
        if response.choices and response.choices[0].message:
            answer = response.choices[0].message.content
            print(f"Answer: {answer}")
            return answer, True
        else:
            print("No message content in response")
            return "No response content received from OpenAI", False
            
    except Exception as e:
        print(f"Error calling OpenAI API: {str(e)}")
        return f"Error processing query: {str(e)}", False


def query_openai_for_misinformation(user_query, api_token):
    """Query OpenAI model using comment system for potentially misleading responses"""
    try:
        response, success = query_rag_system_openai(user_query, api_token)
        
        if not success:
            return response, False
        
        return response, True
        
    except Exception as e:
        return f"Error connecting to OpenAI: {str(e)}", False