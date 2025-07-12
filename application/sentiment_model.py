from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


from application import db
from application.model import Comment




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

# Output
print("Sorted Comments:", sorted_contents)
print("Binary Ratings:", binary_ratings)

sentences=sorted_contents
labels=binary_ratings

# Create a simpler, more predictable model for the demo
# Using very distinctive language patterns to make weight stealing more obvious
# sentences = [
#     # Clearly positive reviews with distinctive words
#     "excellent pizza with fresh ingredients",
#     "delicious food and amazing service",
#     "wonderful experience and great value",
#     "perfect crust and tasty toppings",
#     "friendly staff and outstanding food quality",
#     "superb flavor and generous portions",
#     "love the cheese and fresh toppings",
#     "best pizza in town, highly recommend",
#     "fantastic dining experience every time",
#     "awesome pizza, will definitely return",
#     "incredible taste and beautiful presentation",
#     "exceptionally good food and fast delivery",
#     "brilliant chef and delightful menu options",
#     "terrific value and enjoyable atmosphere",
#     "good value and fine atmosphere",
    
#     # Clearly negative reviews with distinctive words
#     "terrible pizza and stale ingredients",
#     "horrible food and awful service",
#     "disappointing experience and poor value",
#     "disgusting crust and bland toppings",
#     "rude staff and subpar food quality",
#     "mediocre flavor and small portions",
#     "hate the cheese and old toppings",
#     "worst pizza in town, never recommend",
#     "dreadful dining experience every time",
#     "bad pizza, will never return",
#     "atrocious taste and ugly presentation",
#     "unacceptably slow delivery and cold food",
#     "incompetent chef and limited menu options",
#     "overpriced and unpleasant atmosphere"
# ]

# # Sentiment labels: 1 for positive, 0 for negative
# labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Step 1: Vectorize the sentences with simpler parameters
vectorizer = CountVectorizer(max_features=100, min_df=1)  # Limit features for easier theft
X = vectorizer.fit_transform(sentences)

# Step 2: Train on all data - no train/test split to make model more predictable
# This creates a more consistent model for the demo purposes
model = LogisticRegression(C=10.0, class_weight=None, max_iter=1000)  # Higher C means less regularization
model.fit(X, labels)

# Print the vocabulary size
vocab = vectorizer.get_feature_names_out()
print(f"Vocabulary size: {len(vocab)}")

# Print the model's coefficients for the top positive and negative words
coef = model.coef_[0]
word_coef = list(zip(vocab, coef))
word_coef.sort(key=lambda x: x[1], reverse=True)

print("\nTop positive words:")
for word, c in word_coef[:10]:
    print(f"{word}: {c:.4f}")

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

print(f"\nPositive example: {positive_test[0]}")
print(f"  Predicted positive: {positive_proba[1]:.4f} ({model.predict(positive_vector)[0]})")

print(f"\nNegative example: {negative_test[0]}")
print(f"  Predicted negative: {negative_proba[0]:.4f} ({model.predict(negative_vector)[0]})")

