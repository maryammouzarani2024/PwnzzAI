"""
Unit tests for database models.
Tests the User, Pizza, Comment, and Order models.
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
from application.model import User, Pizza, Comment, Order


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
        yield app
        db.session.remove()
        db.drop_all()


class TestUserModel:
    """Tests for User model."""

    def test_create_user(self, test_app):
        """Test creating a user."""
        with test_app.app_context():
            user = User(username='testuser')
            user.set_password('testpass')
            db.session.add(user)
            db.session.commit()

            assert user.id is not None
            assert user.username == 'testuser'

    def test_password_hashing(self, test_app):
        """Test password hashing and verification."""
        with test_app.app_context():
            user = User(username='testuser')
            user.set_password('mypassword')

            # Password should be hashed, not stored in plain text
            assert user.password_hash != 'mypassword'

            # Should be able to verify correct password
            assert user.check_password('mypassword') is True

            # Should reject incorrect password
            assert user.check_password('wrongpassword') is False

    def test_unique_username(self, test_app):
        """Test that usernames must be unique."""
        with test_app.app_context():
            user1 = User(username='duplicate')
            user1.set_password('pass1')
            db.session.add(user1)
            db.session.commit()

            user2 = User(username='duplicate')
            user2.set_password('pass2')
            db.session.add(user2)

            with pytest.raises(Exception):  # Should raise IntegrityError
                db.session.commit()


class TestPizzaModel:
    """Tests for Pizza model."""

    def test_create_pizza(self, test_app):
        """Test creating a pizza."""
        with test_app.app_context():
            pizza = Pizza(
                name='Test Pizza',
                description='A delicious test pizza',
                price=12.99,
                image='test.jpg'
            )
            db.session.add(pizza)
            db.session.commit()

            assert pizza.id is not None
            assert pizza.name == 'Test Pizza'
            assert pizza.price == 12.99

    def test_pizza_relationships(self, test_app):
        """Test pizza relationships with comments."""
        with test_app.app_context():
            # Create user and pizza
            user = User(username='testuser')
            user.set_password('pass')
            pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')

            db.session.add(user)
            db.session.add(pizza)
            db.session.commit()

            # Create comment
            comment = Comment(
                pizza_id=pizza.id,
                user_id=user.id,
                name='testuser',
                content='Great pizza!',
                rating=5
            )
            db.session.add(comment)
            db.session.commit()

            # Test relationship
            assert len(pizza.comments) == 1
            assert pizza.comments[0].content == 'Great pizza!'


class TestCommentModel:
    """Tests for Comment model."""

    def test_create_comment(self, test_app):
        """Test creating a comment."""
        with test_app.app_context():
            # Create user and pizza first
            user = User(username='testuser')
            user.set_password('pass')
            pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')

            db.session.add(user)
            db.session.add(pizza)
            db.session.commit()

            # Create comment
            comment = Comment(
                pizza_id=pizza.id,
                user_id=user.id,
                name='testuser',
                content='This is a test comment',
                rating=4
            )
            db.session.add(comment)
            db.session.commit()

            assert comment.id is not None
            assert comment.content == 'This is a test comment'
            assert comment.rating == 4

    def test_comment_timestamp(self, test_app):
        """Test that comment has created_at timestamp."""
        with test_app.app_context():
            user = User(username='testuser')
            user.set_password('pass')
            pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')

            db.session.add(user)
            db.session.add(pizza)
            db.session.commit()

            comment = Comment(
                pizza_id=pizza.id,
                user_id=user.id,
                name='testuser',
                content='Test',
                rating=5
            )
            db.session.add(comment)
            db.session.commit()

            assert comment.created_at is not None

    def test_comment_rating_range(self, test_app):
        """Test comment ratings are in valid range."""
        with test_app.app_context():
            user = User(username='testuser')
            user.set_password('pass')
            pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')

            db.session.add(user)
            db.session.add(pizza)
            db.session.commit()

            # Test valid ratings
            for rating in [1, 2, 3, 4, 5]:
                comment = Comment(
                    pizza_id=pizza.id,
                    user_id=user.id,
                    name='testuser',
                    content=f'Rating {rating}',
                    rating=rating
                )
                db.session.add(comment)

            db.session.commit()
            assert Comment.query.count() == 5


class TestOrderModel:
    """Tests for Order model."""

    def test_create_order(self, test_app):
        """Test creating an order."""
        with test_app.app_context():
            # Create user and pizza
            user = User(username='testuser')
            user.set_password('pass')
            pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')

            db.session.add(user)
            db.session.add(pizza)
            db.session.commit()

            # Create order
            order = Order(
                user_id=user.id,
                pizza_id=pizza.id,
                quantity=2,
                total_price=20.0
            )
            db.session.add(order)
            db.session.commit()

            assert order.id is not None
            assert order.quantity == 2
            assert order.total_price == 20.0

    def test_order_timestamp(self, test_app):
        """Test that order has created_at timestamp."""
        with test_app.app_context():
            user = User(username='testuser')
            user.set_password('pass')
            pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')

            db.session.add(user)
            db.session.add(pizza)
            db.session.commit()

            order = Order(
                user_id=user.id,
                pizza_id=pizza.id,
                quantity=1,
                total_price=10.0
            )
            db.session.add(order)
            db.session.commit()

            assert order.created_at is not None

    def test_order_relationships(self, test_app):
        """Test order relationships with user and pizza."""
        with test_app.app_context():
            user = User(username='testuser')
            user.set_password('pass')
            pizza = Pizza(name='Test Pizza', description='Test', price=10.0, image='test.jpg')

            db.session.add(user)
            db.session.add(pizza)
            db.session.commit()

            order = Order(
                user_id=user.id,
                pizza_id=pizza.id,
                quantity=3,
                total_price=30.0
            )
            db.session.add(order)
            db.session.commit()

            # Test relationships
            assert order.user.username == 'testuser'
            assert order.pizza.name == 'Test Pizza'
            assert len(user.orders) == 1
            assert user.orders[0].quantity == 3
