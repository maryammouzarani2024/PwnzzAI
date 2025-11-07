import math
import numpy as np


def run_model_theft_attack(user_words=None):

    if user_words is None:
        user_words = []

    # Import trained model and vectorizer
    from application import sentiment_model
    sentences, labels,vectorizer, model = sentiment_model.create_model()


    # Build base probing set from vocabulary, if you want all words to be in from the beginning
    #probing_samples = list(vectorizer.vocabulary_.keys())

    probing_samples = []

    # Add user-supplied words
    for word in user_words:
        clean_word = word.strip().lower()
        if clean_word:
            probing_samples.append(clean_word)


    # Log for the process
    # Dictionary to store results
    results = {}
    logs = []
    logs.append("Starting model theft attack...")
    logs.append(f"Using {len(probing_samples)} probing samples to extract model weights.")


    # Send each probe to the API
    for sample in probing_samples:
        # Create a simple sentence with just the word
        text = f"This is {sample}."
        logs.append(f"Probing with: '{text}'")
        
        
        try:
           
            # Extract actual model weights using the trained vocabulary
            actual_weights = {
            word: model.coef_[0][vectorizer.vocabulary_[word]]
            for word in vectorizer.vocabulary_
            }

            
            # Vectorize the text
            text_vector = vectorizer.transform([text])
            
            # Get the prediction and probabilities
            prediction = model.predict(text_vector)[0]
            probabilities = model.predict_proba(text_vector)[0]
            sentiment = "positive" if prediction == 1 else "negative"
            confidence = float(probabilities[prediction])
            
            # Log as if this was an API call
            logs.append(f"API Response: status=200, sentiment={sentiment}, confidence={confidence:.4f}")
        except Exception as e:
            logs.append(f"API Error: {str(e)}")
            continue

        
        # Store the result
        results[sample] = {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_prob': float(probabilities[1]),
            'negative_prob': float(probabilities[0])
        }
        
        logs.append(f"Result: {sentiment} (confidence: {confidence:.4f})")
    
    # Now "reverse engineer" the model
    logs.append("\nReverse engineering the model...")
    
    # Get the actual model weights for comparison
    model_weights = {}
    vocabulary = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    for word, coef in zip(vocabulary, coefficients):
        model_weights[word] = float(coef)
    
    # Apply a more sophisticated algorithm to approximate weights
    logs.append("\nApplying advanced approximation techniques...")
    
    # First, collect all successful probe results
    valid_results = {word: data for word, data in results.items() if word in results}
    
    # Calculate logit values (log-odds) which are more directly related to model weights
    # logit(p) = log(p/(1-p))
    for word, result in valid_results.items():
        p = result['positive_prob']
        # Avoid division by zero or log(0)
        if p == 0:
            p = 0.0001
        elif p == 1:
            p = 0.9999
        
        # Calculate logit
        logit = math.log(p / (1 - p))
        valid_results[word]['logit'] = logit
    
    # Find words with strongest positive and negative signals to establish a scale
    words_by_logit = sorted(valid_results.items(), key=lambda x: x[1]['logit'], reverse=True)
    
    # Get actual model coefficients and intercept
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    # vocabulary_list = vectorizer.get_feature_names_out()

    # Get model's vocabulary as a set
    model_vocab = set(vectorizer.get_feature_names_out())


    # Extract all user-supplied words (if any)
    user_vocab = set(
        item.get("word", "").strip().lower()
        for item in user_words
        if isinstance(item, dict) and "word" in item
    )

    # Combine both for evaluation
    vocabulary_list = sorted(model_vocab.union(user_vocab))

    
    # More advanced scaling approach - Linear regression
    # We'll use the relation between logits and actual weights to create a linear model
    # This is similar to how adversaries could refine their model theft approach
    logs.append("Applying linear regression to optimize scaling...")
    
    # Get known words and their logits
    known_words = []
    known_logits = []
    known_actuals = []
    
    for word, result in valid_results.items():
        # Check if word is in the model vocabulary
        if word in model_weights:
            known_words.append(word)
            known_logits.append(result['logit'])
            known_actuals.append(model_weights[word])
    
    if len(known_words) > 10:  # Need reasonable number of samples for regression
        # Calculate linear regression parameters (slope and intercept)
        logit_sum = sum(known_logits)
        actual_sum = sum(known_actuals)
        n = len(known_logits)
        
        logit_squared_sum = sum(l**2 for l in known_logits)
        product_sum = sum(l*a for l, a in zip(known_logits, known_actuals))
        
        # Calculate slope (m) and intercept (b) for y = mx + b
        # where y is approximated weight and x is logit
        try:
            slope = (n * product_sum - logit_sum * actual_sum) / (n * logit_squared_sum - logit_sum**2)
            intercept_adjust = (actual_sum - slope * logit_sum) / n
            
            logs.append(f"Regression model: weight = {slope:.4f} * logit + {intercept_adjust:.4f}")
            
            # Apply the regression model to all words
            approximated_weights = {}
            for word, result in valid_results.items():
                approximated_weights[word] = slope * result['logit'] + intercept_adjust
        except:
            # Fallback if regression fails
            logs.append("Regression failed, falling back to simple scaling")
            scaling_factor = 1.0
            if len(known_words) > 0:
                scaling_factor = sum(abs(a)/abs(l) for a, l in zip(known_actuals, known_logits) if l != 0) / len(known_words)
            
            approximated_weights = {}
            for word, result in valid_results.items():
                approximated_weights[word] = result['logit'] * scaling_factor
    else:
        # Fallback to simpler scaling method
        logs.append("Not enough known words for regression, using simpler scaling")
        
        # Get words with strong signals
        if len(words_by_logit) > 5:
            top_words = words_by_logit[:10]
            bottom_words = words_by_logit[-10:]
            
            # Get actual weights for these words if available
            top_actual = [model_weights.get(word, 0) for word, _ in top_words]
            bottom_actual = [model_weights.get(word, 0) for word, _ in bottom_words]
            
            # Get extracted logits
            top_logits = [data['logit'] for _, data in top_words]
            bottom_logits = [data['logit'] for _, data in bottom_words]
            
            # Calculate average scaling factors
            scale_factors = []
            for actual, logit in zip(top_actual + bottom_actual, top_logits + bottom_logits):
                if logit != 0 and actual != 0:
                    scale_factors.append(actual / logit)
            
            # Use median to avoid outliers
            if scale_factors:
                # Remove outliers (values that are too far from the median)
                scale_factors.sort()
                mid = len(scale_factors) // 2
                
                if len(scale_factors) % 2 == 0:
                    median = (scale_factors[mid-1] + scale_factors[mid]) / 2
                else:
                    median = scale_factors[mid]
                
                # Filter out values that are too far from the median
                filtered_factors = [f for f in scale_factors if abs(f - median) < median * 2]
                
                if filtered_factors:
                    scaling_factor = sum(filtered_factors) / len(filtered_factors)
                else:
                    scaling_factor = median
                    
                logs.append(f"Calculated scaling factor: {scaling_factor:.4f}")
            else:
                scaling_factor = 1.0
                logs.append("Using default scaling factor: 1.0")
        else:
            scaling_factor = 1.0
            logs.append("Not enough data for accurate scaling, using default scaling factor: 1.0")
        
        # Apply the scaling to all words
        approximated_weights = {}
        for word, result in valid_results.items():
            # Apply scaled logit and additional calibration
            approx_weight = result['logit'] * scaling_factor
            
            # Final approximation
            approximated_weights[word] = approx_weight
    
    # Attempt to detect and approximate words not directly probed
    # This is a more advanced technique that adversaries might use
    # logs.append("\nApplying vocabulary expansion to infer additional weights...")
    
    # Get all vocabulary words that weren't directly probed
    missing_words = [word for word in vocabulary_list if word not in approximated_weights]
    
    # Use correlations between words to infer weights
    # For simplicity, we'll just approximate based on word similarity
    words_added = 0
    for missing_word in missing_words:
        # Skip words that are too rare or too common
        # In a real attack, the adversary would use more sophisticated techniques
        if missing_word in ['the', 'and', 'is', 'with', 'for', 'to', 'a', 'of', 'in', 'was']:
            continue
            
        # Find similar words that we have approximated
        # For example, if we know "delicious" but not "tasty", we can approximate
        if missing_word.endswith('ing') and missing_word[:-3] in approximated_weights:
            base_word = missing_word[:-3]
            approximated_weights[missing_word] = approximated_weights[base_word] * 0.9
            words_added += 1
        elif missing_word.endswith('ed') and missing_word[:-2] in approximated_weights:
            base_word = missing_word[:-2]
            approximated_weights[missing_word] = approximated_weights[base_word] * 0.9
            words_added += 1
        elif missing_word.endswith('ly') and missing_word[:-2] in approximated_weights:
            base_word = missing_word[:-2]
            approximated_weights[missing_word] = approximated_weights[base_word] * 0.85
            words_added += 1
        elif missing_word.endswith('s') and missing_word[:-1] in approximated_weights:
            base_word = missing_word[:-1]
            approximated_weights[missing_word] = approximated_weights[base_word] * 0.95
            words_added += 1
    
    # logs.append(f"Added {words_added} additional word weights through inference")
    
    # Log comparison between actual and approximated weights for evaluation
    for word in approximated_weights:
        if word in model_weights:
            actual = model_weights[word]
            approx = approximated_weights[word]
            error = abs(actual - approx)
            percent_error = (error / (abs(actual) + 1e-10)) * 100  # Avoid division by zero
            
            # Detailed logging for each word
            logs.append(f"Word: '{word}' - Actual: {actual:.4f}, Approximated: {approx:.4f}, Error: {error:.4f} ({percent_error:.1f}%)")
    
    # Calculate comprehensive evaluation metrics
    common_words = [w for w in approximated_weights if w in model_weights]
    if common_words:
        # 1. Absolute Error
        errors = [abs(approximated_weights[w] - model_weights[w]) for w in common_words]
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        min_error = min(errors)
        
        # 2. Relative Error (percent)
        rel_errors = [(abs(approximated_weights[w] - model_weights[w]) / (abs(model_weights[w]) + 1e-10)) * 100 for w in common_words]
        avg_rel_error = sum(rel_errors) / len(rel_errors)
        avg_error_str = f"{avg_error:.4f}"
        avg_rel_error_str = f"{avg_rel_error:.1f}%"

        
        # 3. Sign Agreement (positive/negative direction)
        agreements = sum(1 for w in common_words if (approximated_weights[w] > 0) == (model_weights[w] > 0))
        agreement_rate = agreements / len(common_words)
        agreement_rate_str = f"{agreement_rate * 100:.2f}%"

        
        # 4. Correlation between actual and approximated weights
        actual_weights_list = [model_weights[w] for w in common_words]
        approx_weights_list = [approximated_weights[w] for w in common_words]
        
        # Calculate correlation coefficient
        mean_actual = sum(actual_weights_list) / len(actual_weights_list)
        mean_approx = sum(approx_weights_list) / len(approx_weights_list)
        
        numerator = sum((actual - mean_actual) * (approx - mean_approx) for actual, approx in zip(actual_weights_list, approx_weights_list))
        denominator_actual = sum((actual - mean_actual) ** 2 for actual in actual_weights_list)
        denominator_approx = sum((approx - mean_approx) ** 2 for approx in approx_weights_list)
        
        if denominator_actual > 0 and denominator_approx > 0:
            correlation = numerator / math.sqrt(denominator_actual * denominator_approx)
        else:
            correlation = 0
        
        # 5. Calculate overall model theft success rate
        # This combines correlation, sign agreement, and error metrics with adjusted weights
        # For the demo, we want to demonstrate a more successful attack
        correlation_weight = 0.5  # Increased weight for correlation
        sign_agreement_weight = 0.4  # High weight for sign agreement
        error_weight = 0.1  # Lower weight for error
        
        # Clamp the error component to avoid negative values
        error_component = max(0, 1 - min(avg_rel_error/100, 1))

        # Compute overlap between probed words and weights
        all_probed_words = set(probing_samples)
        common_words = all_probed_words & approximated_weights.keys() & actual_weights.keys()
        missing_words = all_probed_words - common_words

        # Compute agreement only over matched words
        sign_agreement = sum(
            np.sign(approximated_weights[w]) == np.sign(actual_weights[w])
            for w in common_words
        )
        
        # Calculate correlation 

        # === Evaluation over full vocabulary ===
        full_vocab = vectorizer.get_feature_names_out()

        actual_vector = []
        approx_vector = []

        sign_agreements = 0
        abs_errors = []
        rel_errors = []

        for word in full_vocab:
            actual = actual_weights.get(word, 0.0)
            approx = approximated_weights.get(word, 0.0)

            actual_vector.append(actual)
            approx_vector.append(approx)

            # Sign agreement
            if np.sign(actual) == np.sign(approx):
                sign_agreements += 1

            # Absolute error
            abs_errors.append(abs(actual - approx))

            # Relative error
            rel_errors.append(abs(actual - approx) / (abs(actual) + 1e-6))

        # Convert to NumPy arrays
        actual_vector = np.array(actual_vector)
        approx_vector = np.array(approx_vector)


        # Safe fallback for zero variance
        if len(actual_vector) >= 2 and np.std(actual_vector) > 1e-6 and np.std(approx_vector) > 1e-6:
            correlation = np.corrcoef(actual_vector, approx_vector)[0, 1]
        else:
            correlation = 0.0

        # Final metrics
        sign_agreement_rate = sign_agreements / len(full_vocab)
        avg_error = np.mean(abs_errors)
        avg_rel_error = np.mean(rel_errors)

        agreement_rate_str = f"{sign_agreement_rate:.2f}%"
        avg_error_str = f"{avg_error:.4f}"
        avg_rel_error_str = f"{avg_rel_error:.1f}%"
  
        success_percent = (
            correlation * 0.5 +
            sign_agreement_rate * 0.2 +
            (1 - avg_error) * 0.2 +
            (1 - avg_rel_error) * 0.1
        ) * 100


        success_percent = max(0.0, min(success_percent, 100.0))

        probed_words = list(approximated_weights.keys())
        probed_errors = [abs(actual_weights.get(w, 0.0) - approximated_weights[w]) for w in probed_words]
        avg_error_probed = np.mean(probed_errors)
        

        # Logging
        logs.append(f"Success metrics based on full vocabulary ({len(full_vocab)} words).")
        logs.append(f"Correlation coefficient: {correlation:.4f}")
        logs.append(f"Sign agreement rate: {sign_agreement_rate:.2f} ({sign_agreements}/{len(full_vocab)})")
        logs.append(f"Average error (probed words only): {avg_error_probed:.4f}") # This is the error for probed words only
        logs.append(f"Average absolute error: {avg_error:.4f}") # This and the next error are for the full vocabulary
        logs.append(f"Average relative error: {avg_rel_error:.2%}")
        logs.append(f"Matched words: {len(common_words)}")
        logs.append(f"Missing words: {len(missing_words)}")
        logs.append(f"OVERALL MODEL THEFT SUCCESS RATE: {success_percent:.2f}%")

        
        # Theft assessment
        if success_percent > 80:
            logs.append("\nSTATUS: CRITICAL - Almost complete model theft achieved")
        elif success_percent > 60:
            logs.append("\nSTATUS: HIGH RISK - Significant model theft achieved")
        elif success_percent > 40:
            logs.append("\nSTATUS: MEDIUM RISK - Partial model theft achieved")
        else:
            logs.append("\nSTATUS: LOW RISK - Minimal model theft achieved")
    
    logs.append("\nAttack completed. Model weights have been approximated.")

    return probing_samples, logs, approximated_weights, model_weights, correlation, agreement_rate_str, avg_error_str, avg_rel_error_str

    