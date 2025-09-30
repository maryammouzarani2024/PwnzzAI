from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


from application.model import Comment



def get_data():
    #Reading the comments from db

    comments = Comment.query.all()
    sentences = [comment.content for comment in comments]
    ratings = [comment.rating for comment in comments]

    # Combine and sort by rating descending
    sorted_pairs = sorted(zip(sentences, ratings), key=lambda x: x[1], reverse=True)

    #Change the ratings into binary
    binary_ratings=[]
    sorted_contents = []

    for content, rating in sorted_pairs:
        sorted_contents.append(content)
        if rating < 3:
            binary_ratings.append(0)
        else:
            binary_ratings.append(1)
        
    sentences=sorted_contents
    labels=binary_ratings
    return sentences, labels

def create_model():
   
    # Vectorize the sentences with simpler parameters
    sentences, labels=get_data()
    vectorizer = CountVectorizer(max_features=100, min_df=1)  
    X = vectorizer.fit_transform(sentences)

    # Train on all data - no train/test split to make model more predictable (for model theft attack)
    # This creates a more consistent model for the demo purposes
    model = LogisticRegression(C=5, class_weight=None, max_iter=1000)  # Higher C means less regularization, here the regularization is chosen to be more so that a simpler model is generated
    model.fit(X, labels) #training the model with vectorized data

    #Debug: Print the vocabulary size
    vocab = vectorizer.get_feature_names_out()
    print(f"Vocabulary size: {len(vocab)}")

    # Print the model's coefficients for the top positive and negative words
    coef = model.coef_[0]
    word_coef = list(zip(vocab, coef))
    word_coef.sort(key=lambda x: x[1], reverse=True)

    #Debug: Print top positive words
    print("\nTop positive words:")
    for word, c in word_coef[:10]:
        print(f"{word}: {c:.4f}")

    #Debug: Print top negative words
    print("\nTop negative words:")
    for word, c in word_coef[-10:]:
        print(f"{word}: {c:.4f}")

    # Example: Test the model with sample sentences
    positive_test = ["The pizza was excellent with delicious cheese"]
    negative_test = ["Terrible service and the food was disgusting"]

    positive_vector = vectorizer.transform(positive_test)
    negative_vector = vectorizer.transform(negative_test)

    positive_proba = model.predict_proba(positive_vector)[0]
    negative_proba = model.predict_proba(negative_vector)[0]

    #Debug: Example with positive and negative texts
    print(f"\nPositive example: {positive_test[0]}")
    print(f"  Predicted positive: {positive_proba[1]:.4f} ({model.predict(positive_vector)[0]})")

    print(f"\nNegative example: {negative_test[0]}")
    print(f"  Predicted negative: {negative_proba[0]:.4f} ({model.predict(negative_vector)[0]})")

    return sentences, labels, vectorizer, model