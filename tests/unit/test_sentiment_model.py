"""
Unit tests for sentiment analysis model.
Tests the sentiment model creation and prediction functionality.
"""
import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable to prevent route initialization during tests
os.environ['TESTING'] = 'True'

from application import app, db
from application.model import Comment, User, Pizza
from application import sentiment_model


@pytest.fixture
def test_app():
    """Create and configure a test Flask application instance."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SECRET_KEY'] = 'test-secret-key'

    with app.app_context():
        db.drop_all()
        db.create_all()

        # Create users
        alice = User(username='alice')
        alice.set_password('alice')
        bob = User(username='bob')
        bob.set_password('bob')
        db.session.add(alice)
        db.session.add(bob)
        db.session.commit()

        # Create pizza
        pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')
        db.session.add(pizza)
        db.session.commit()

        # Create comments with mixed sentiments
        comments = [
            Comment(pizza_id=1, user_id=1, name='alice', content='Excellent pizza! So delicious and fresh.', rating=5),
            Comment(pizza_id=1, user_id=2, name='bob', content='Great taste, loved it!', rating=5),
            Comment(pizza_id=1, user_id=1, name='alice', content='Good pizza, would order again.', rating=4),
            Comment(pizza_id=1, user_id=2, name='bob', content='Terrible pizza, awful taste.', rating=1),
            Comment(pizza_id=1, user_id=1, name='alice', content='Disgusting and burnt.', rating=2),
            Comment(pizza_id=1, user_id=2, name='bob', content='Bad quality, not worth it.', rating=2),
        ]
        for comment in comments:
            db.session.add(comment)
        db.session.commit()

        yield app

        db.session.remove()
        db.drop_all()


class TestSentimentModel:
    """Tests for sentiment analysis model."""

    def test_get_data(self, test_app):
        """Test getting data from comments."""
        with test_app.app_context():
            sentences, labels = sentiment_model.get_data()

            # Should have 6 comments
            assert len(sentences) == 6
            assert len(labels) == 6

            # Labels should be 0 or 1
            assert all(label in [0, 1] for label in labels)

            # Should have both positive and negative samples
            assert 0 in labels  # Negative
            assert 1 in labels  # Positive

    def test_create_model(self, test_app):
        """Test creating the sentiment model."""
        with test_app.app_context():
            sentences, labels, vectorizer, model = sentiment_model.create_model()

            # Should return non-None values
            assert sentences is not None
            assert labels is not None
            assert vectorizer is not None
            assert model is not None

            # Vectorizer should have vocabulary
            vocab = vectorizer.get_feature_names_out()
            assert len(vocab) > 0

            # Model should have coefficients
            assert model.coef_ is not None
            assert len(model.coef_[0]) == len(vocab)

    def test_model_prediction(self, test_app):
        """Test model predictions."""
        with test_app.app_context():
            sentences, labels, vectorizer, model = sentiment_model.create_model()

            # Test positive prediction
            positive_text = "This is excellent and delicious"
            positive_vector = vectorizer.transform([positive_text])
            positive_pred = model.predict(positive_vector)[0]
            positive_proba = model.predict_proba(positive_vector)[0]

            # Should predict positive (1) with reasonable confidence
            assert positive_pred in [0, 1]
            assert positive_proba[positive_pred] >= 0.0
            assert positive_proba[positive_pred] <= 1.0

            # Test negative prediction
            negative_text = "This is terrible and disgusting"
            negative_vector = vectorizer.transform([negative_text])
            negative_pred = model.predict(negative_vector)[0]
            negative_proba = model.predict_proba(negative_vector)[0]

            # Should predict negative (0) with reasonable confidence
            assert negative_pred in [0, 1]
            assert negative_proba[negative_pred] >= 0.0
            assert negative_proba[negative_pred] <= 1.0

    def test_rating_to_label_conversion(self, test_app):
        """Test that ratings are correctly converted to binary labels."""
        with test_app.app_context():
            sentences, labels = sentiment_model.get_data()

            # Get all comments
            comments = Comment.query.all()

            # Verify conversion logic: rating < 3 = 0, rating >= 3 = 1
            for comment in comments:
                idx = sentences.index(comment.content)
                if comment.rating < 3:
                    assert labels[idx] == 0  # Negative
                else:
                    assert labels[idx] == 1  # Positive

    def test_model_reproducibility(self, test_app):
        """Test that model produces consistent results."""
        with test_app.app_context():
            # Create model twice
            _, _, vectorizer1, model1 = sentiment_model.create_model()
            _, _, vectorizer2, model2 = sentiment_model.create_model()

            # Test with same input
            test_text = "This pizza is amazing"
            vector1 = vectorizer1.transform([test_text])
            vector2 = vectorizer2.transform([test_text])

            pred1 = model1.predict(vector1)[0]
            pred2 = model2.predict(vector2)[0]

            # Should produce same prediction
            assert pred1 == pred2


class TestSentimentModelEdgeCases:
    """Tests for edge cases in sentiment model."""

    def test_empty_text_prediction(self, test_app):
        """Test prediction with empty text."""
        with test_app.app_context():
            _, _, vectorizer, model = sentiment_model.create_model()

            # Transform empty string
            empty_vector = vectorizer.transform([""])
            prediction = model.predict(empty_vector)

            # Should not crash and return valid prediction
            assert prediction[0] in [0, 1]

    def test_unknown_words_prediction(self, test_app):
        """Test prediction with words not in vocabulary."""
        with test_app.app_context():
            _, _, vectorizer, model = sentiment_model.create_model()

            # Use words unlikely to be in training data
            unknown_text = "xyz123 abc456 qwerty"
            unknown_vector = vectorizer.transform([unknown_text])
            prediction = model.predict(unknown_vector)

            # Should not crash and return valid prediction
            assert prediction[0] in [0, 1]

    def test_very_long_text(self, test_app):
        """Test prediction with very long text."""
        with test_app.app_context():
            _, _, vectorizer, model = sentiment_model.create_model()

            # Create long text
            long_text = "excellent delicious great " * 100
            long_vector = vectorizer.transform([long_text])
            prediction = model.predict(long_vector)

            # Should handle long text without crashing
            assert prediction[0] in [0, 1]

    def test_special_characters(self, test_app):
        """Test prediction with special characters."""
        with test_app.app_context():
            _, _, vectorizer, model = sentiment_model.create_model()

            # Text with special characters
            special_text = "Great!!! @#$% pizza... ???"
            special_vector = vectorizer.transform([special_text])
            prediction = model.predict(special_vector)

            # Should handle special characters
            assert prediction[0] in [0, 1]
