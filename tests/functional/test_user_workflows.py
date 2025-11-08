"""
Functional tests for complete user workflows.
Tests end-to-end user scenarios from registration to order completion.
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

        # Create test users
        alice = User(username='alice')
        alice.set_password('alice123')
        bob = User(username='bob')
        bob.set_password('bob123')
        db.session.add(alice)
        db.session.add(bob)
        db.session.commit()

        # Create test pizzas
        pizzas = [
            Pizza(name='Margherita', description='Classic cheese pizza', price=9.99, image='margherita.jpg'),
            Pizza(name='Pepperoni', description='Pepperoni pizza', price=11.99, image='pepperoni.jpg'),
            Pizza(name='Veggie Supreme', description='Loaded with veggies', price=12.99, image='veggie.jpg'),
        ]
        for pizza in pizzas:
            db.session.add(pizza)
        db.session.commit()

        yield app

        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return test_app.test_client()


class TestCompleteUserJourney:
    """Test complete user journey from login to checkout."""

    def test_full_ordering_workflow(self, client, test_app):
        """Test complete workflow: login -> browse -> comment -> order -> view orders."""
        with test_app.app_context():
            # Step 1: User logs in
            response = client.post('/login', data={
                'username': 'alice',
                'password': 'alice123'
            }, follow_redirects=True)

            assert response.status_code == 200
            assert b'pizza' in response.data.lower()

            # Step 2: User browses pizzas
            response = client.get('/')
            assert response.status_code == 200
            assert b'Margherita' in response.data or b'margherita' in response.data.lower()

            # Step 3: User views pizza details
            response = client.get('/pizza/1')
            assert response.status_code == 200

            # Step 4: User adds a comment
            response = client.post('/add_comment/1', data={
                'content': 'Absolutely delicious pizza!',
                'rating': '5'
            }, follow_redirects=True)

            assert response.status_code == 200
            assert b'delicious' in response.data.lower()

            # Step 5: User places an order
            response = client.post('/order/1', data={
                'quantity': '2'
            }, follow_redirects=True)

            assert response.status_code == 200

            # Step 6: User views their orders
            response = client.get('/orders')
            assert response.status_code == 200

            # Verify order was created
            orders = Order.query.filter_by(user_id=1).all()
            assert len(orders) == 1
            assert orders[0].quantity == 2
            assert orders[0].pizza_id == 1

            # Verify comment was created
            comments = Comment.query.filter_by(user_id=1).all()
            assert len(comments) == 1
            assert comments[0].content == 'Absolutely delicious pizza!'
            assert comments[0].rating == 5

    def test_multiple_orders_workflow(self, client, test_app):
        """Test user ordering multiple different pizzas."""
        with test_app.app_context():
            # Login
            client.post('/login', data={
                'username': 'alice',
                'password': 'alice123'
            })

            # Order first pizza
            client.post('/order/1', data={'quantity': '2'})

            # Order second pizza
            client.post('/order/2', data={'quantity': '1'})

            # Order third pizza
            client.post('/order/3', data={'quantity': '3'})

            # Check orders
            orders = Order.query.filter_by(user_id=1).all()
            assert len(orders) == 3

            # Verify total quantities
            total_items = sum(order.quantity for order in orders)
            assert total_items == 6

    def test_comment_edit_workflow(self, client, test_app):
        """Test user adding and deleting comments."""
        with test_app.app_context():
            # Login
            client.post('/login', data={
                'username': 'alice',
                'password': 'alice123'
            })

            # Add first comment
            client.post('/add_comment/1', data={
                'content': 'Great pizza!',
                'rating': '5'
            })

            # Add second comment
            client.post('/add_comment/1', data={
                'content': 'Even better the second time!',
                'rating': '5'
            })

            # Verify two comments exist
            comments = Comment.query.filter_by(user_id=1).all()
            assert len(comments) == 2

            # Delete first comment
            first_comment_id = comments[0].id
            client.post(f'/delete_comment/{first_comment_id}')

            # Verify only one comment remains
            comments = Comment.query.filter_by(user_id=1).all()
            assert len(comments) == 1
            assert comments[0].content == 'Even better the second time!'


class TestMultiUserInteractions:
    """Test interactions between multiple users."""

    def test_multiple_users_ordering_same_pizza(self, client, test_app):
        """Test multiple users ordering the same pizza."""
        with test_app.app_context():
            # Alice logs in and orders
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            client.post('/order/1', data={'quantity': '2'})
            client.get('/logout')

            # Bob logs in and orders
            client.post('/login', data={'username': 'bob', 'password': 'bob123'})
            client.post('/order/1', data={'quantity': '3'})

            # Verify both orders exist
            orders = Order.query.filter_by(pizza_id=1).all()
            assert len(orders) == 2

            # Verify order ownership
            alice_orders = Order.query.filter_by(user_id=1).all()
            bob_orders = Order.query.filter_by(user_id=2).all()
            assert len(alice_orders) == 1
            assert len(bob_orders) == 1
            assert alice_orders[0].quantity == 2
            assert bob_orders[0].quantity == 3

    def test_users_commenting_on_same_pizza(self, client, test_app):
        """Test multiple users commenting on the same pizza."""
        with test_app.app_context():
            # Alice comments
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            client.post('/add_comment/1', data={
                'content': 'Love this pizza!',
                'rating': '5'
            })
            client.get('/logout')

            # Bob comments
            client.post('/login', data={'username': 'bob', 'password': 'bob123'})
            client.post('/add_comment/1', data={
                'content': 'Not my favorite.',
                'rating': '3'
            })

            # Verify both comments exist
            comments = Comment.query.filter_by(pizza_id=1).all()
            assert len(comments) == 2

            # Verify comment ownership
            alice_comments = Comment.query.filter_by(user_id=1).all()
            bob_comments = Comment.query.filter_by(user_id=2).all()
            assert len(alice_comments) == 1
            assert len(bob_comments) == 1

    def test_user_cannot_delete_others_comments(self, client, test_app):
        """Test that users cannot delete other users' comments."""
        with test_app.app_context():
            # Alice adds a comment
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            client.post('/add_comment/1', data={
                'content': 'My comment',
                'rating': '5'
            })
            alice_comment = Comment.query.filter_by(user_id=1).first()
            alice_comment_id = alice_comment.id
            client.get('/logout')

            # Bob tries to delete Alice's comment
            client.post('/login', data={'username': 'bob', 'password': 'bob123'})
            response = client.post(f'/delete_comment/{alice_comment_id}', follow_redirects=True)

            # Comment should still exist
            comment = Comment.query.get(alice_comment_id)
            assert comment is not None
            assert comment.user_id == 1

            # Should see error message
            assert b'own' in response.data.lower() or b'cannot' in response.data.lower()


class TestAuthenticationFlows:
    """Test authentication-related workflows."""

    def test_login_logout_cycle(self, client, test_app):
        """Test complete login/logout cycle."""
        with test_app.app_context():
            # Login
            response = client.post('/login', data={
                'username': 'alice',
                'password': 'alice123'
            }, follow_redirects=True)
            assert response.status_code == 200

            # Access protected resource
            response = client.get('/orders')
            assert response.status_code == 200

            # Logout
            response = client.get('/logout', follow_redirects=True)
            assert response.status_code == 200

            # Try to access protected resource (should redirect)
            response = client.get('/orders', follow_redirects=True)
            assert b'login' in response.data.lower() or b'log in' in response.data.lower()

    def test_invalid_login_attempts(self, client, test_app):
        """Test handling of invalid login attempts."""
        with test_app.app_context():
            # Wrong password
            response = client.post('/login', data={
                'username': 'alice',
                'password': 'wrongpassword'
            }, follow_redirects=True)
            assert b'invalid' in response.data.lower() or b'error' in response.data.lower()

            # Non-existent user
            response = client.post('/login', data={
                'username': 'nonexistent',
                'password': 'password'
            }, follow_redirects=True)
            assert b'invalid' in response.data.lower() or b'error' in response.data.lower()

    def test_protected_routes_require_auth(self, client, test_app):
        """Test that protected routes require authentication."""
        with test_app.app_context():
            # Try to access protected routes without login
            protected_routes = [
                ('/orders', 'GET'),
                ('/order/1', 'POST', {'quantity': '1'}),
                ('/add_comment/1', 'POST', {'content': 'test', 'rating': '5'}),
            ]

            for route_info in protected_routes:
                if len(route_info) == 2:
                    route, method = route_info
                    data = None
                else:
                    route, method, data = route_info

                if method == 'GET':
                    response = client.get(route, follow_redirects=True)
                else:
                    response = client.post(route, data=data, follow_redirects=True)

                # Should redirect to login or show login message
                assert response.status_code == 200
                assert b'login' in response.data.lower() or b'log in' in response.data.lower()


class TestBusinessLogic:
    """Test business logic and validation."""

    def test_order_price_calculation(self, client, test_app):
        """Test that order prices are calculated correctly."""
        with test_app.app_context():
            # Login
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})

            # Get pizza price
            pizza = Pizza.query.get(1)
            pizza_price = pizza.price

            # Place order
            quantity = 3
            client.post('/order/1', data={'quantity': str(quantity)})

            # Verify order total
            order = Order.query.filter_by(user_id=1).first()
            expected_total = pizza_price * quantity
            assert abs(order.total_price - expected_total) < 0.01  # Account for floating point

    def test_order_quantity_validation(self, client, test_app):
        """Test order quantity validation."""
        with test_app.app_context():
            # Login
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})

            # Try to order 0 quantity
            response = client.post('/order/1', data={'quantity': '0'}, follow_redirects=True)
            assert b'quantity' in response.data.lower() or b'error' in response.data.lower()

            # Try negative quantity
            response = client.post('/order/1', data={'quantity': '-1'}, follow_redirects=True)
            assert response.status_code in [200, 400]

            # Verify no orders were created
            orders = Order.query.filter_by(user_id=1).all()
            assert len(orders) == 0

    def test_comment_rating_validation(self, client, test_app):
        """Test comment rating validation."""
        with test_app.app_context():
            # Login
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})

            # Add comments with valid ratings (1-5)
            for rating in [1, 2, 3, 4, 5]:
                client.post('/add_comment/1', data={
                    'content': f'Rating {rating} test',
                    'rating': str(rating)
                })

            # Verify all comments were created
            comments = Comment.query.filter_by(user_id=1).all()
            assert len(comments) == 5


class TestDataConsistency:
    """Test data consistency and integrity."""

    def test_order_history_persistence(self, client, test_app):
        """Test that order history persists across sessions."""
        with test_app.app_context():
            # Session 1: Place orders
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            client.post('/order/1', data={'quantity': '2'})
            client.post('/order/2', data={'quantity': '1'})
            client.get('/logout')

            # Session 2: Check orders persist
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            orders = Order.query.filter_by(user_id=1).all()
            assert len(orders) == 2

    def test_comment_persistence(self, client, test_app):
        """Test that comments persist and are visible to all users."""
        with test_app.app_context():
            # Alice adds comment
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            client.post('/add_comment/1', data={
                'content': 'Excellent pizza!',
                'rating': '5'
            })
            client.get('/logout')

            # Bob should see Alice's comment
            client.post('/login', data={'username': 'bob', 'password': 'bob123'})
            response = client.get('/pizza/1')
            assert b'Excellent pizza!' in response.data

    def test_user_isolation(self, client, test_app):
        """Test that user data is properly isolated."""
        with test_app.app_context():
            # Alice creates data
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            client.post('/order/1', data={'quantity': '2'})
            client.get('/logout')

            # Bob logs in
            client.post('/login', data={'username': 'bob', 'password': 'bob123'})

            # Bob should not see Alice's orders
            response = client.get('/orders')
            alice_orders = Order.query.filter_by(user_id=1).all()
            bob_orders = Order.query.filter_by(user_id=2).all()

            assert len(alice_orders) == 1
            assert len(bob_orders) == 0


class TestErrorHandling:
    """Test error handling in various scenarios."""

    def test_nonexistent_pizza_handling(self, client, test_app):
        """Test handling of requests for non-existent pizzas."""
        with test_app.app_context():
            # Try to view non-existent pizza
            response = client.get('/pizza/9999')
            assert response.status_code == 404

            # Try to order non-existent pizza
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})
            response = client.post('/order/9999', data={'quantity': '1'})
            assert response.status_code == 404

    def test_missing_form_data_handling(self, client, test_app):
        """Test handling of missing form data."""
        with test_app.app_context():
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})

            # Try to add comment without content
            response = client.post('/add_comment/1', data={
                'rating': '5'
            }, follow_redirects=True)
            assert response.status_code == 200

            # Try to order without quantity
            response = client.post('/order/1', data={}, follow_redirects=True)
            assert response.status_code in [200, 400]

    def test_session_expiration_handling(self, client, test_app):
        """Test handling of expired or invalid sessions."""
        with test_app.app_context():
            # Login
            client.post('/login', data={'username': 'alice', 'password': 'alice123'})

            # Clear session to simulate expiration
            with client.session_transaction() as sess:
                sess.clear()

            # Try to access protected resource
            response = client.get('/orders', follow_redirects=True)
            assert b'login' in response.data.lower() or b'log in' in response.data.lower()
