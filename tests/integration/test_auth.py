"""
Integration tests for authentication endpoints.
Tests login, logout, and session management.
"""
import pytest
import json


class TestAuthentication:
    """Tests for user authentication."""

    def test_login_page_loads(self, client):
        """Test that login page loads successfully."""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'login' in response.data.lower()

    def test_login_with_valid_credentials(self, client):
        """Test login with valid username and password."""
        response = client.post(
            '/login',
            data={'username': 'alice', 'password': 'alice'},
            follow_redirects=True
        )

        assert response.status_code == 200
        # Check if redirected to home page
        assert b'pizza' in response.data.lower()

    def test_login_with_invalid_username(self, client):
        """Test login with non-existent username."""
        response = client.post(
            '/login',
            data={'username': 'nonexistent', 'password': 'password'},
            follow_redirects=True
        )

        assert response.status_code == 200
        assert b'invalid' in response.data.lower() or b'error' in response.data.lower()

    def test_login_with_invalid_password(self, client):
        """Test login with wrong password."""
        response = client.post(
            '/login',
            data={'username': 'alice', 'password': 'wrongpassword'},
            follow_redirects=True
        )

        assert response.status_code == 200
        assert b'invalid' in response.data.lower() or b'error' in response.data.lower()

    def test_logout(self, authenticated_client):
        """Test logout functionality."""
        response = authenticated_client.get('/logout', follow_redirects=True)

        assert response.status_code == 200
        # Should be redirected to home or login
        assert b'pizza' in response.data.lower() or b'login' in response.data.lower()


class TestSessionManagement:
    """Tests for session management and protected routes."""

    def test_session_persists_after_login(self, client):
        """Test that session data is maintained after login."""
        # Login
        client.post(
            '/login',
            data={'username': 'alice', 'password': 'alice'}
        )

        # Access a page that uses session
        response = client.get('/orders')
        assert response.status_code == 200

    def test_protected_route_without_login(self, client):
        """Test accessing protected route without authentication."""
        response = client.get('/orders')

        # Should redirect to login or show error
        assert response.status_code in [302, 200]
        if response.status_code == 200:
            assert b'login' in response.data.lower() or b'error' in response.data.lower()

    def test_add_comment_requires_login(self, client):
        """Test that adding comment requires authentication."""
        response = client.post(
            '/add_comment/1',
            data={'name': 'Test', 'content': 'Great pizza!', 'rating': '5'},
            follow_redirects=True
        )

        assert response.status_code == 200
        # Should show login requirement message
        assert b'login' in response.data.lower() or b'log in' in response.data.lower()

    def test_order_pizza_requires_login(self, client):
        """Test that ordering requires authentication."""
        response = client.post(
            '/order/1',
            data={'quantity': '1'},
            follow_redirects=True
        )

        assert response.status_code == 200
        # Should show login requirement
        assert b'login' in response.data.lower() or b'log in' in response.data.lower()


class TestLabSetup:
    """Tests for lab setup API endpoints."""

    def test_save_openai_api_key(self, client):
        """Test saving OpenAI API key to session."""
        response = client.post(
            '/save-openai-api-key',
            json={'api_key': 'sk-test123456789'}
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is True

    def test_save_openai_api_key_invalid_format(self, client):
        """Test saving API key with invalid format."""
        response = client.post(
            '/save-openai-api-key',
            json={'api_key': 'invalid-key'}
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is False
        assert 'error' in data

    def test_save_openai_api_key_empty(self, client):
        """Test saving empty API key."""
        response = client.post(
            '/save-openai-api-key',
            json={'api_key': ''}
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['success'] is False

    def test_check_openai_api_key_not_set(self, client):
        """Test checking API key when not set."""
        response = client.get('/check-openai-api-key')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'has_key' in data
        assert data['has_key'] is False

    def test_check_openai_api_key_after_setting(self, client):
        """Test checking API key after setting it."""
        # Set API key
        client.post(
            '/save-openai-api-key',
            json={'api_key': 'sk-test123456789'}
        )

        # Check if it's set
        response = client.get('/check-openai-api-key')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['has_key'] is True

    def test_check_ollama_status(self, client):
        """Test checking Ollama status endpoint."""
        response = client.get('/check-ollama-status')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'available' in data
        assert 'models' in data
        assert isinstance(data['models'], list)
