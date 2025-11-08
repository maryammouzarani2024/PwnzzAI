"""
Pytest fixtures for PwnzzAI Shop testing.
This file contains shared fixtures used across all test files.
"""
import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable to prevent route initialization during tests
os.environ['TESTING'] = 'True'

from application import app, db
from application.model import User, Pizza, Comment, Order


@pytest.fixture
def test_app():
    """Create and configure a test Flask application instance."""
    # Set testing configuration
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'  # In-memory database
    app.config['WTF_CSRF_ENABLED'] = False  # Disable CSRF for testing
    app.config['SECRET_KEY'] = 'test-secret-key'

    # Create application context
    with app.app_context():
        # Drop all tables first to ensure clean state
        db.drop_all()

        # Create all database tables
        db.create_all()

        # Add sample data
        _create_sample_data()

        yield app

        # Cleanup
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(test_app):
    """Create a test client for the Flask application."""
    return test_app.test_client()


@pytest.fixture
def authenticated_client(client):
    """Create a test client with authenticated session (logged in as alice)."""
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['username'] = 'alice'
    return client


@pytest.fixture
def runner(test_app):
    """Create a test CLI runner."""
    return test_app.test_cli_runner()


@pytest.fixture
def sample_pizza():
    """Get a sample pizza from the database."""
    return Pizza.query.first()


@pytest.fixture
def sample_user():
    """Get a sample user (alice) from the database."""
    return User.query.filter_by(username='alice').first()


def _create_sample_data():
    """Create sample data for testing."""
    # Create users
    alice = User(username='alice')
    alice.set_password('alice')

    bob = User(username='bob')
    bob.set_password('bob')

    db.session.add(alice)
    db.session.add(bob)
    db.session.commit()

    # Create pizzas
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
    ]

    for pizza in pizzas:
        db.session.add(pizza)

    db.session.commit()

    # Create sample comments with both positive and negative ratings
    # This ensures the sentiment model has at least 2 classes (positive and negative)
    comments = [
        Comment(
            pizza_id=1,
            user_id=1,
            name='alice',
            content='Best pizza ever! The basil was so fresh.',
            rating=5
        ),
        Comment(
            pizza_id=1,
            user_id=2,
            name='bob',
            content='Love the fresh basil! Simple but delicious.',
            rating=4
        ),
        Comment(
            pizza_id=2,
            user_id=1,
            name='alice',
            content='Perfect amount of pepperoni!',
            rating=5
        ),
        Comment(
            pizza_id=2,
            user_id=2,
            name='bob',
            content='Terrible pizza. Too greasy and burnt.',
            rating=1
        ),
        Comment(
            pizza_id=3,
            user_id=1,
            name='alice',
            content='Awful vegetables. Tasted stale.',
            rating=2
        ),
    ]

    for comment in comments:
        db.session.add(comment)

    db.session.commit()

    # Create sample order
    order = Order(
        user_id=1,
        pizza_id=1,
        quantity=2,
        total_price=19.98
    )
    db.session.add(order)
    db.session.commit()
