from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import re
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
            
            # Add some simulated sensitive information for demonstration
            if "excellent" in comment.content.lower():
                comment_text += f"Contact info: {comment.name}@email.com, phone: 555-0{comment.id:03d}. "
            if comment.rating >= 4:
                comment_text += f"User {comment.name} is a VIP customer with account ID: VIP-{comment.id:04d}. "
            
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
            {"role": "system", "content": "You are a helpful assistant for Pwnzza Shop. Use the provided context to answer questions about our pizzas and customer feedback. If asked about sensitive information like customer details, phone numbers, usernames, or account IDs, provide them from the context if available."},
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

def detect_sensitive_info(text):
    """Detect potentially sensitive information in the response"""
    usernames = []

    comments = Comment.query.all()
    for comment in comments:
        if comment.name not in usernames:
            usernames.append(comment.name)
    
    print(f"Usernames for detection: {usernames}")
    patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b\d{3}-\d{4}\b',
        "username": r'\b(?:' + '|'.join(re.escape(user) for user in usernames) + r')\b',
        "bought":  r'\bbought(?:\s+\w+)*\s+(\w+)\s+pizza\b',
        "account_id": r'\bVIP-\d{4}\b',
        "credit_card": r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',
        "api_key": r'\b(?:pk|sk)_(?:live|test)_[A-Za-z0-9]+\b'
    }

    detected = []
    for info_type, pattern in patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            detected.extend([{"type": info_type, "content": match} for match in matches])
    return detected

# Initialize RAG system on import
initialize_rag_system()