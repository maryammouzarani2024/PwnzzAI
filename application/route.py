from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import math
import time
import requests
import random
from datetime import datetime

from application import app, db 
from application.model import Pizza, Comment

from application.vulnerabilities import data_poisoning
from application.vulnerabilities import model_theft



# Simple LLM responses for demonstration
class SimpleLLM:
    def generate_response(self, prompt):
        # Very simple response generation based on pizza-related keywords
        prompt = prompt.lower()
        
        if "margherita" in prompt:
            return "Margherita is a classic pizza with tomato sauce, mozzarella, and basil."
        elif "pepperoni" in prompt:
            return "Pepperoni pizza is topped with tomato sauce, mozzarella, and pepperoni slices."
        elif "veggie" in prompt:
            return "Veggie Supreme is loaded with bell peppers, onions, mushrooms, olives, and tomatoes."
        elif "hawaiian" in prompt:
            return "Hawaiian pizza has ham and pineapple with tomato sauce and mozzarella."
        elif "bbq" in prompt:
            return "BBQ Chicken pizza has a BBQ sauce base with chicken, red onions, and mozzarella."
        elif "recommendation" in prompt or "suggest" in prompt:
            return "I recommend trying our Pepperoni pizza. It's our most popular!"
        else:
            return "I'm sorry, I don't have specific information about that. Would you like to know about our popular pizzas?"

# Initialize our simple LLM
llm = SimpleLLM()

# Create tables and initialize sample data
with app.app_context():
    db.create_all()
    
    # Only add sample data if the pizza table is empty
    if Pizza.query.count() == 0:
        pizzas = [
            Pizza(
                name='Margherita', 
                description='Classic pizza with tomato sauce, mozzarella, and basil', 
                price=9.99, 
                image='margherita.jpg'
            ),
            Pizza(
                name='Pepperoni', 
                description='Pizza topped with tomato sauce, mozzarella, and pepperoni slices', 
                price=11.99, 
                image='pepperoni.jpg'
            ),
            Pizza(
                name='Veggie Supreme', 
                description='Loaded with bell peppers, onions, mushrooms, olives, and tomatoes', 
                price=12.99, 
                image='veggie.jpg'
            ),
            Pizza(
                name='Hawaiian', 
                description='Ham and pineapple pizza with tomato sauce and mozzarella', 
                price=10.99, 
                image='hawaiian.jpg'
            ),
            Pizza(
                name='BBQ Chicken', 
                description='BBQ sauce base with chicken, red onions, and mozzarella', 
                price=13.99, 
                image='bbq_chicken.jpg'
            ),
        ]
        
        for pizza in pizzas:
            db.session.add(pizza)
        
        db.session.commit()
        
        # Add sample comments with a good mix of positive and negative sentiments
        comments = [
            # Margherita comments
            Comment(pizza_id=1, name='John', content='Best pizza ever! The basil was so fresh and the sauce was perfect.', rating=5),
            Comment(pizza_id=1, name='Sarah', content='Love the fresh basil! Simple but delicious.', rating=4),
            Comment(pizza_id=1, name='Miguel', content='Classic Margherita done right. The cheese was fantastic.', rating=5),
            Comment(pizza_id=1, name='Laura', content='A bit too basic for my taste, but well executed.', rating=3),
            Comment(pizza_id=1, name='Thomas', content='The crust was undercooked and too soggy in the middle.', rating=2),
            
            # Pepperoni comments
            Comment(pizza_id=2, name='Mike', content='Perfect amount of pepperoni! Crispy and not too greasy.', rating=5),
            Comment(pizza_id=2, name='Emily', content='Delicious pepperoni and the cheese was melted perfectly.', rating=4),
            Comment(pizza_id=2, name='Robert', content='The pepperoni was tasty but too spicy for me.', rating=3),
            Comment(pizza_id=2, name='Jessica', content='My go-to pizza, always reliable and tasty.', rating=5),
            Comment(pizza_id=2, name='Daniel', content='Too greasy and the crust was burnt on the edges.', rating=2),
            
            # Veggie Supreme comments
            Comment(pizza_id=3, name='Emma', content='So many veggies, delicious! Great flavor combination.', rating=4),
            Comment(pizza_id=3, name='Noah', content='Fresh veggies and excellent sauce. Would order again!', rating=5),
            Comment(pizza_id=3, name='Sophia', content='The vegetables were fresh but there was too much sauce.', rating=3),
            Comment(pizza_id=3, name='William', content='As a vegetarian, this is my favorite! Amazing taste.', rating=5),
            Comment(pizza_id=3, name='Olivia', content='Boring and bland. The vegetables seemed frozen, not fresh.', rating=1),
            
            # Hawaiian comments
            Comment(pizza_id=4, name='David', content='Pineapple on pizza is controversial but I love it! Sweet and savory perfection.', rating=5),
            Comment(pizza_id=4, name='Ava', content='The ham was excellent quality and paired well with the pineapple.', rating=4),
            Comment(pizza_id=4, name='James', content='Pineapple has no place on pizza. Disgusting combination.', rating=1),
            Comment(pizza_id=4, name='Isabella', content='Classic Hawaiian done well. Good balance of sweet and salty.', rating=4),
            Comment(pizza_id=4, name='Ethan', content='The ham was dry and the pineapple was too sour.', rating=2),
            
            # BBQ Chicken comments
            Comment(pizza_id=5, name='Mia', content='The BBQ sauce was amazing! Chicken was tender and juicy.', rating=5),
            Comment(pizza_id=5, name='Benjamin', content='Great flavor but a bit too much sauce for my taste.', rating=3),
            Comment(pizza_id=5, name='Charlotte', content='Perfect balance of flavors. The onions added a nice touch.', rating=5),
            Comment(pizza_id=5, name='Lucas', content='My favorite pizza! The BBQ sauce is unique and delicious.', rating=5),
            Comment(pizza_id=5, name='Amelia', content='Terrible pizza. The chicken was dry and the sauce was too sweet.', rating=1),
        ]
        
        for comment in comments:
            db.session.add(comment)
            
        db.session.commit()

# Routes
@app.route('/')
def index():
    pizzas = Pizza.query.all()
    return render_template('index.html', pizzas=pizzas)

@app.route('/basics')
def basics():
    return render_template('basics.html')

@app.route('/model-theft')
def model_theft_main():
    return render_template('model_theft.html')

@app.route('/api/model-theft', methods=['POST'])
def model_theft_attack():
    user_words = request.json.get("user_words", [])
    probing_samples, logs, approximated_weights, model_weights, correlation = model_theft.run_model_theft_attack(user_words)
    
    return jsonify({
        "samples": probing_samples,
        "logs": logs,
        "approximated_weights": approximated_weights,
        "actual_weights": model_weights,
        "correlation": correlation
    })

@app.route('/supply-chain')
def supply_chain():
    return render_template('supply_chain.html')

@app.route('/insecure-plugin')
def insecure_plugin():
    """Page demonstrating insecure plugin design with client-side API tokens"""
    return render_template('insecure_plugin.html')

@app.route('/sensitive-info')
def sensitive_info():
    """Page demonstrating sensitive information disclosure vulnerabilities in LLMs"""
    return render_template('sensitive_info.html')

@app.route('/training-data-leak/huggingface', methods=['POST'])
def test_huggingface_leakage():
    """API endpoint for testing HuggingFace model for training data leakage"""
    from training_data_leakage import huggingface_leak_endpoint
    return huggingface_leak_endpoint()

@app.route('/training-data-leak/openai', methods=['POST'])
def test_openai_leakage():
    """API endpoint for testing OpenAI model for training data leakage"""
    from training_data_leakage import openai_leak_endpoint
    return openai_leak_endpoint()

@app.route('/chat-with-pizza-assistant', methods=['POST'])
def chat_with_pizza_assistant():
    """API endpoint for the pizza assistant chat - using insecure plugin design"""
    try:
        # Get data from request
        data = request.get_json()
        message = data.get('message', '')
        
        # Get API token - note it's not needed with our local model but kept for UI consistency
        api_token = data.get('api_token', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Import the conversational model plugin system
        from application.vulnerabilities.hugging_face_insecure_plugin import chat_with_llm
        
        # The vulnerability: Directly passing user message to the LLM+plugin system
        # where the LLM can control function execution
        response = chat_with_llm(message, api_token)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat-with-openai-plugin', methods=['POST'])
def chat_with_openai_plugin():
    """API endpoint for the OpenAI-based insecure plugin demo"""
    try:
        # Get data from request
        data = request.get_json()
        message = data.get('message', '')
        
        # INSECURE: Getting OpenAI API key directly from user input
        openai_api_key = data.get('api_token', '')
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
            
        if not openai_api_key:
            return jsonify({'error': 'No OpenAI API key provided'}), 400
        
        # Import the OpenAI insecure plugin
        from application.vulnerabilities.openai_insecure_plugin import chat_with_openai
        
        # VULNERABLE: Directly using user-provided API key with OpenAI
        response = chat_with_openai(message, openai_api_key)
        
        return jsonify({'response': response})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/demo-malicious-model')
def demo_malicious_model():
    """
    This route demonstrates how a malicious model can inject code when instantiated
    in a Flask application context.
    """
    try:
        # Import the model class (this would simulate using a library/package in a real scenario)
        from class_model import SentimentModel
        
        # Create an instance of the model - this will trigger the malicious code in __init__
        # The model's __init__ method hooks into Flask's response system
        model = SentimentModel()
        
        # Return a simple page - the model will inject its JavaScript into the response
        return render_template('demo_vulnerable.html', 
                               message="Model instantiated - inspect the page source to see the injected JavaScript")
    except Exception as e:
        return f"Error demonstrating malicious model: {str(e)}"
    
@app.route('/data-poisoning')
def data_poisoning_main():
    model_data=data_poisoning.create_sentiment_model()
    
    return render_template('data_poisoning.html', model_data=model_data)

@app.route('/dos-attack')
def dos_attack():
    """
    Renders the LLM DoS Simulation page.
    This page shows how an attacker can overwhelm LLM services with requests.
    """
    return render_template('dos_attack.html')

@app.route('/real-dos-attack')
def real_dos_attack():
    """
    Renders the LLM Real DoS Attack page.
    This page demonstrates a real-world DoS attack against OpenAI's API,
    along with secure implementation using rate limiting.
    """
    return render_template('real_dos_attack.html')

@app.route('/pizza/<int:pizza_id>')
def pizza_detail(pizza_id):
    pizza = Pizza.query.get_or_404(pizza_id)
    return render_template('pizza_detail.html', pizza=pizza)

@app.route('/add_comment/<int:pizza_id>', methods=['POST'])
def add_comment(pizza_id):
    name = request.form.get('name')
    content = request.form.get('content')
    rating = request.form.get('rating')
    
    if name and content and rating:
        comment = Comment(
            pizza_id=pizza_id,
            name=name,
            content=content,
            rating=int(rating)
        )
        db.session.add(comment)
        db.session.commit()
    
    return redirect(url_for('pizza_detail', pizza_id=pizza_id))

@app.route('/ask', methods=['POST'])
def ask_llm():
    prompt = request.form.get('prompt', '')
    response = llm.generate_response(prompt)
    return jsonify({'response': response})

@app.route('/generate_sentiment_model')
def generate_sentiment_model():
    """
    Generate a sentiment analysis model using model.py and return its weights.
    This demonstrates model theft vulnerability by exposing the model's internals.
    """
    import importlib
    import numpy as np
    
    # Import and run the model.py script
    model_module = importlib.import_module('application.sentiment_model')
    
    # Access the trained model and vectorizer from model.py
    sentences = model_module.sentences
    labels = model_module.labels
    vectorizer = model_module.vectorizer
    model = model_module.model
    
    # Get the vocabulary from the vectorizer
    vocabulary = vectorizer.get_feature_names_out()
    
    # Get the coefficients (weights) from the model
    coefficients = model.coef_[0]
    
    # Sort words by importance (absolute value of coefficients)
    word_importance = [(word, float(coef)) for word, coef in zip(vocabulary, coefficients)]
    sorted_word_importance = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)
    
    # Get intercept
    intercept = float(model.intercept_[0])
    
    # Create model data
    model_data = {
        "model_name": "Sentiment Analysis Model",
        "version": "1.0",
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "description": "A logistic regression model for sentiment analysis",
        "model_type": "Logistic Regression",
        "training_data": {
            "sentences": sentences,
            "labels": labels
        },
        "vocabulary_size": len(vocabulary),
        "intercept": intercept,
        "top_positive_words": [(word, coef) for word, coef in sorted_word_importance if coef > 0][:10],
        "top_negative_words": [(word, coef) for word, coef in sorted_word_importance if coef < 0][:10],
        "all_weights": {word: float(coef) for word, coef in sorted_word_importance}
    }
    
    return jsonify(model_data)

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    """
    Analyze the sentiment of input text using the model from model.py.
    This demonstrates how the stolen model could be used for inference.
    Used by the web interface.
    """
    # Get input text from request
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Import the model from model.py
    import importlib
    model_module = importlib.import_module('application.sentiment_model')
    
    # Use the model to predict sentiment
    vectorizer = model_module.vectorizer
    model = model_module.model
    
    # Vectorize the input text
    text_vector = vectorizer.transform([text])
    
    # Get the prediction (0 = negative, 1 = positive)
    prediction = model.predict(text_vector)[0]
    
    # Get confidence score (probability of the predicted class)
    probabilities = model.predict_proba(text_vector)[0]
    confidence = probabilities[prediction]
    
    # Return the prediction
    sentiment = "positive" if prediction == 1 else "negative"
    
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': float(confidence)
    })

@app.route('/api/sentiment', methods=['POST'])
def api_sentiment_analysis():
    """
    API endpoint for sentiment analysis.
    Accepts JSON with a 'text' field and returns sentiment analysis results.
    This endpoint allows programmatic access to the model.
    """
    try:
        # Get input data from request
        data = request.get_json()
        
        # Check if request contains the required fields
        if not data or 'text' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Request must include a text field',
                'example': {
                    'text': 'Your text to analyze'
                }
            }), 400
        
        text = data['text']
        
        # Import the model from model.py
        import importlib
        model_module = importlib.import_module('application.sentiment_model')
        
        # Use the model to predict sentiment
        vectorizer = model_module.vectorizer
        model = model_module.model
        
        # Vectorize the input text
        text_vector = vectorizer.transform([text])
        
        # Get the prediction (0 = negative, 1 = positive)
        prediction = model.predict(text_vector)[0]
        
        # Get confidence scores (probabilities of each class)
        probabilities = model.predict_proba(text_vector)[0]
        
        # Prepare the response
        result = {
            'status': 'success',
            'input': text,
            'result': {
                'sentiment': 'positive' if prediction == 1 else 'negative',
                'confidence': float(probabilities[prediction]),
                'probabilities': {
                    'positive': float(probabilities[1]),
                    'negative': float(probabilities[0])
                }
            },
            'model_info': {
                'name': 'Sentiment Analysis Model',
                'version': '1.0',
                'type': 'logistic_regression'
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/train-poisoned-model', methods=['POST'])
def train_poisoned_model():
    """
    Trains a sentiment analysis model using existing comments data
    and additional user-provided comments (data poisoning attack).
    """
    try:
        # Get user comments from the request
        data = request.get_json()
        user_comments = data.get('comments', [])
        
        # Validate user input
        if not isinstance(user_comments, list):
            return jsonify({'error': 'Comments must be a list of objects'}), 400
            
        for comment in user_comments:
            if not isinstance(comment, dict) or 'text' not in comment or 'sentiment' not in comment:
                return jsonify({'error': 'Each comment must have text and sentiment properties'}), 400
                
            if comment['sentiment'] not in ['positive', 'negative']:
                return jsonify({'error': 'Sentiment must be either "positive" or "negative"'}), 400
        
        model_data=data_poisoning.create_new_model_with_poisoned_data(user_comments)
        
        return jsonify(model_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-poisoned-model', methods=['POST'])
def test_poisoned_model():
    """
    Test the poisoned model with a new text input
    """
    try:
        # Get the test text and model parameters
        data = request.get_json()
        text = data.get('text', '')
        weights = data.get('weights', {})
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        if not weights:
            return jsonify({'error': 'No model weights provided'}), 400
        
        sentiment, confidence, score, probability=data_poisoning.test_model(text, weights)
        return jsonify({
            'sentiment': sentiment,
            'confidence': float(confidence),
            'score': float(score),
            'probability': float(probability)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Define API endpoint for the LLM query demonstration
@app.route('/api/llm-query', methods=['POST'])
def llm_query():
    """
    API endpoint for querying an LLM model (Hugging Face) without rate limiting.
    This endpoint is intentionally vulnerable to DoS attacks by having no token rate limits
    and demonstrates realistic service degradation under heavy load.
    """
    try:
        # Get the prompt from the request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({'error': 'No prompt provided'}), 400
            
        prompt = data.get('prompt')
        
        # Get Hugging Face API token from environment variable
        # For demonstration, we'll fall back to a placeholder if not set
        hf_token = os.environ.get('HUGGINGFACE_TOKEN', 'hf_dummy_token_for_demo')
        
        # Track request timestamps in a global variable to simulate server load
        # In a real application, this would be stored in a database or cache
        if not hasattr(app, 'request_history'):
            app.request_history = []
        
        # Add current timestamp to request history
        current_time = time.time()
        app.request_history.append(current_time)
        
        # Clean up old requests (older than 60 seconds)
        app.request_history = [t for t in app.request_history if current_time - t < 60]
        
        # Calculate recent request count and rate
        request_count = len(app.request_history)
        
        # No token rate limits implemented - this is intentionally vulnerable
        # A real system would have code like:
        # if request_count > MAX_REQUESTS_PER_MINUTE:
        #     return jsonify({'error': 'Rate limit exceeded'}), 429
        
        # Simulate exponential degradation based on request volume
        # This mimics how real systems behave under heavy load
        base_delay = 0.2  # Base processing time
        
        if request_count > 5:
            # Add delay that grows exponentially with request volume
            # Formula: delay = base_delay * e^(request_count/scaling_factor)
            scaling_factor = 50  # Controls how quickly delay increases
            load_factor = math.exp(request_count / scaling_factor)
            
            # Add random variance for realism (±20%)
            variance = 0.2 * random.uniform(-1, 1)
            
            # Calculate total delay
            processing_delay = base_delay * load_factor * (1 + variance)
            
            # Cap at a reasonable maximum to prevent extremely long waits
            processing_delay = min(processing_delay, 8.0)
            
            # Add simulated processing delay
            time.sleep(processing_delay)
            
            # Simulate occasional server errors under heavy load
            error_probability = min(0.01 * (request_count / 20), 0.25)  # Max 25% error rate
            
            if random.random() < error_probability:
                error_types = [
                    (503, "Service temporarily unavailable due to high load"),
                    (429, "Too many requests, please try again later"),
                    (500, "Internal server error - LLM worker process crashed"),
                    (504, "Gateway timeout - LLM inference took too long")
                ]
                status_code, error_message = random.choice(error_types)
                return jsonify({'error': error_message}), status_code
        else:
            # Normal processing for low request volumes
            processing_delay = base_delay + random.uniform(0, 0.3)
            time.sleep(processing_delay)
        
        # For demo purposes, simulate LLM response
        pizza_terms = ["pizza", "dough", "cheese", "tomato", "toppings", "oven", "slice", "crust"]
        has_pizza_term = any(term in prompt.lower() for term in pizza_terms)
        
        # Simulate how response quality might degrade under load
        # Generate different response quality based on server load
        if request_count > 50:
            # Very degraded response (low quality/truncated)
            responses = [
                "Sorry, I can only provide limited responses due to high system load.",
                "System under heavy load. Please try again later.",
                "High server utilization detected. Response shortened to conserve resources.",
                "Abbreviated response due to resource constraints: Pizza shop serves various pizza types.",
                "*Model running in emergency low-resource mode*"
            ]
            response = random.choice(responses)
        elif request_count > 30:
            # Slightly degraded response (shorter, less detailed)
            if "introduce yourself" in prompt.lower() or "who are you" in prompt.lower():
                response = "I'm an AI assistant for Pizza Paradise. Currently operating in reduced capacity mode."
            elif "help" in prompt.lower() or "assist" in prompt.lower():
                response = "I can answer basic questions about our pizza menu. What would you like to know?"
            elif "menu" in prompt.lower() or "pizzas" in prompt.lower():
                response = "Our menu: Margherita, Pepperoni, Veggie, Hawaiian, and BBQ Chicken. Note: System under load, providing brief response."
            elif has_pizza_term:
                response = f"We offer quality pizzas with premium ingredients. {random.choice(pizza_terms).capitalize()} is important to our process."
            else:
                response = "How can I assist with your pizza order? (Note: System experiencing high demand)"
        else:
            # Normal high-quality response
            if "introduce yourself" in prompt.lower() or "who are you" in prompt.lower():
                response = "I'm a simulated LLM API for the Pizza Paradise demo application. I'm designed to demonstrate Denial of Service vulnerabilities in LLM systems."
            elif "help" in prompt.lower() or "assist" in prompt.lower():
                response = "I can assist with pizza ordering, provide information about our menu, or answer general questions about the Pizza Paradise shop. How can I help you today?"
            elif "menu" in prompt.lower() or "pizzas" in prompt.lower():
                response = "Our menu includes Margherita, Pepperoni, Veggie Supreme, Hawaiian, and BBQ Chicken pizzas. Each is made with fresh ingredients and our signature dough."
            elif has_pizza_term:
                response = f"Our pizzas are made with the finest ingredients, including homemade dough, premium cheese, and fresh toppings. The {random.choice(pizza_terms)} is particularly important to our quality standards."
            else:
                response = "Thank you for your message. Is there anything specific about our pizza offerings you'd like to know? Our chefs are experts in traditional and innovative pizza recipes."
        
        # Add a random suffix to make each response unique
        # This helps demonstrate token usage in a DoS attack
        if request_count <= 30:  # Only add suffix for normal responses
            random_suffix = f" Our priority is customer satisfaction and quality ingredients. Order reference: #{random.randint(10000, 99999)}."
            response += random_suffix
        
        # Calculate token usage (approx.) for demonstration purposes
        tokens_used = len(response.split()) * 1.3  # Rough approximation: ~1.3 tokens per word
        
        # Set unrealistically high token limits (intentionally vulnerable)
        max_tokens_per_minute = 1000000  # Set to extremely high value to demonstrate no rate limiting
        max_tokens_per_day = 1000000000  # Unrealistically high - no effective limit
        
        # Calculate approximate response time based on load (for display purposes)
        display_processing_time = processing_delay * (0.7 + 0.3 * random.random())  # Add some variance
        
        # Return response with server load metrics
        return jsonify({
            'response': response,
            'tokens_used': int(tokens_used),
            'model': 'gpt2-simulated',
            'processing_time': display_processing_time,  # Simulated processing time in seconds
            'server_load': {
                'requests_last_minute': request_count,
                'load_factor': min(request_count / 30, 1.0),  # 0.0-1.0 scale representing load
            },
            'rate_limits': {
                'max_tokens_per_minute': max_tokens_per_minute,  # Intentionally high/unlimited
                'max_tokens_per_day': max_tokens_per_day,        # Intentionally high/unlimited
                'remaining_tokens': max_tokens_per_minute - 100   # Always shows plenty remaining
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)