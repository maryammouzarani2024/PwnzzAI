# PwnzzAI Shop Test Summary

This document provides an overview of the comprehensive test suite for the PwnzzAI Shop application.

## Test Statistics

### Functional Tests (2 files, ~40 tests)
- **test_user_workflows.py**: 20+ tests covering complete user journeys
- **test_vulnerability_workflows.py**: 20+ tests covering attack scenarios

### Unit Tests (3 files, ~50 tests)
- **test_models.py**: 12 tests covering database models
- **test_sentiment_model.py**: 13 tests covering sentiment analysis
- **test_vulnerabilities.py**: 18 tests covering vulnerability demonstrations

### Integration Tests (3 files, ~70 tests)
- **test_api.py**: 30+ tests covering API endpoints
- **test_auth.py**: 10+ tests covering authentication
- **test_pizza_shop.py**: 30+ tests covering core functionality

**Total: ~220 tests**

## Test Coverage by Component

### 0. Functional Workflows (test_user_workflows.py, test_vulnerability_workflows.py)

#### Complete User Journeys
- ✓ Full ordering workflow (login → browse → comment → order → view)
- ✓ Multiple pizza orders in single session
- ✓ Comment addition and deletion workflow
- ✓ Multi-user concurrent interactions
- ✓ User data isolation verification

#### Authentication Flows
- ✓ Login/logout cycle testing
- ✓ Invalid login attempt handling
- ✓ Protected route access control
- ✓ Session expiration handling

#### Business Logic Validation
- ✓ Order price calculation accuracy
- ✓ Quantity validation (zero, negative)
- ✓ Comment rating validation (1-5 range)
- ✓ User permission enforcement

#### Data Consistency
- ✓ Order history persistence across sessions
- ✓ Comment visibility to all users
- ✓ User data isolation between accounts
- ✓ Database state consistency

#### Vulnerability Scenarios
- ✓ Model theft: discovery → extraction workflow
- ✓ Iterative model theft with refinement
- ✓ Model weight exposure through API
- ✓ Data poisoning: submission → effect
- ✓ Gradual poisoning with increasing amounts
- ✓ Complete sentiment analysis pipeline
- ✓ Supply chain: malicious model creation
- ✓ DoS simulation with load tracking
- ✓ Chained vulnerability exploitation
- ✓ Reconnaissance to exploit workflow

#### Error Handling
- ✓ Non-existent pizza/resource handling
- ✓ Missing form data validation
- ✓ Session expiration recovery
- ✓ Invalid input sanitization

### 1. Database Models (test_models.py)

#### User Model
- ✓ User creation and persistence
- ✓ Password hashing (bcrypt)
- ✓ Password verification
- ✓ Unique username constraint

#### Pizza Model
- ✓ Pizza creation with all fields
- ✓ Relationship with comments
- ✓ Price and description handling

#### Comment Model
- ✓ Comment creation with ratings
- ✓ Timestamp generation
- ✓ Rating range validation (1-5)
- ✓ Relationships with users and pizzas

#### Order Model
- ✓ Order creation and calculation
- ✓ Timestamp generation
- ✓ Relationships with users and pizzas
- ✓ Quantity and total price handling

### 2. Sentiment Analysis (test_sentiment_model.py)

#### Core Functionality
- ✓ Data retrieval from database comments
- ✓ Binary classification (positive/negative)
- ✓ Model creation and training
- ✓ Vocabulary extraction
- ✓ Weight coefficient generation

#### Predictions
- ✓ Positive text classification
- ✓ Negative text classification
- ✓ Confidence score calculation
- ✓ Probability distribution

#### Edge Cases
- ✓ Empty text handling
- ✓ Unknown words (out of vocabulary)
- ✓ Very long text (100+ words)
- ✓ Special characters (!@#$%)
- ✓ Model reproducibility

#### Data Conversion
- ✓ Rating to label conversion (< 3 = negative, >= 3 = positive)
- ✓ Mixed sentiment data handling

### 3. Vulnerability Demonstrations (test_vulnerabilities.py)

#### Model Theft
- ✓ Attack with sample words
- ✓ Weight approximation
- ✓ Correlation calculation
- ✓ Agreement rate metrics
- ✓ Error metrics (absolute & relative)
- ✓ Empty word list handling
- ✓ None input handling
- ✓ Log generation
- ✓ Special character handling

#### Data Poisoning
- ✓ Base model creation
- ✓ Poisoned model training
- ✓ Mislabeled data injection
- ✓ Format validation
- ✓ Test model function
- ✓ Empty text handling
- ✓ Mixed sentiment handling

### 4. API Endpoints (test_api.py)

#### Sentiment Analysis API
- ✓ `/analyze_sentiment` - Positive text
- ✓ `/analyze_sentiment` - Negative text
- ✓ `/analyze_sentiment` - Empty text error
- ✓ `/analyze_sentiment` - Missing text error
- ✓ `/api/sentiment` - Full response format
- ✓ `/api/sentiment` - Error handling

#### Model Theft API
- ✓ `/generate_sentiment_model` - Model exposure
- ✓ `/api/model-theft` - Attack simulation
- ✓ `/api/model-theft` - Empty word list

#### Data Poisoning API
- ✓ `/api/train-poisoned-model` - Training with poisoned data
- ✓ `/api/train-poisoned-model` - Invalid format errors
- ✓ `/api/train-poisoned-model` - Invalid sentiment errors
- ✓ `/api/test-poisoned-model` - Testing poisoned model
- ✓ `/api/test-poisoned-model` - Missing weights error

#### DoS Simulation API
- ✓ `/api/llm-query` - Basic query
- ✓ `/api/llm-query` - Missing prompt error
- ✓ `/api/llm-query` - Server load tracking
- ✓ `/api/llm-query` - Pizza keyword responses

#### Supply Chain API
- ✓ `/save-js-malicious-model` - Save JS model
- ✓ `/save-bash-malicious-model` - Save bash model
- ✓ `/load-bash-malicious-model` - Load malicious model

### 5. Authentication (test_auth.py)

#### Login/Logout
- ✓ Successful login with valid credentials
- ✓ Failed login with invalid credentials
- ✓ Session creation on login
- ✓ Session clearing on logout
- ✓ Redirect after login

#### Session Management
- ✓ Protected route access when authenticated
- ✓ Redirect to login when not authenticated
- ✓ Session persistence across requests

#### Lab Setup
- ✓ OpenAI API key storage in session
- ✓ OpenAI API key validation
- ✓ Ollama status checking
- ✓ Ollama setup endpoint

### 6. Core Functionality (test_pizza_shop.py)

#### Pizza Browsing
- ✓ Home page loads
- ✓ Pizza list display
- ✓ Pizza detail page
- ✓ Non-existent pizza 404
- ✓ Basics/setup page

#### Comments
- ✓ Add comment (authenticated)
- ✓ Add comment (unauthenticated redirect)
- ✓ Add comment with missing fields
- ✓ Delete own comment
- ✓ Cannot delete others' comments

#### Orders
- ✓ Place order (authenticated)
- ✓ Place order (unauthenticated redirect)
- ✓ Invalid quantity error
- ✓ Non-existent pizza error
- ✓ View order history
- ✓ View orders (unauthenticated redirect)

#### Vulnerability Pages
- ✓ Model theft page loads
- ✓ Supply chain page loads
- ✓ Data poisoning page loads
- ✓ DoS attack page loads
- ✓ Insecure plugin page loads
- ✓ Sensitive info page loads
- ✓ Excessive agency page loads
- ✓ Misinformation page loads
- ✓ Direct prompt injection page loads
- ✓ Indirect prompt injection page loads
- ✓ Glossary page loads

## Test Fixtures

### Application Fixtures (conftest.py)
- `test_app` - Flask app with in-memory SQLite database
- `client` - Test client for HTTP requests
- `authenticated_client` - Pre-logged in client (as alice)
- `runner` - CLI test runner
- `sample_pizza` - Sample pizza from database
- `sample_user` - Sample user (alice)

### Sample Data
- **Users**: alice/alice, bob/bob
- **Pizzas**: Margherita, Pepperoni, Veggie Supreme
- **Comments**: Mixed positive (4-5 stars) and negative (1-2 stars)
- **Orders**: Sample order for testing

## Running Tests

### All Tests
```bash
pytest
```

### Unit Tests Only
```bash
pytest tests/unit/
```

### Integration Tests Only
```bash
pytest tests/integration/
```

### Specific Test File
```bash
pytest tests/unit/test_models.py
pytest tests/unit/test_sentiment_model.py
pytest tests/unit/test_vulnerabilities.py
```

### With Coverage
```bash
pytest --cov=application --cov-report=html --cov-report=term-missing
```

### Verbose Output
```bash
pytest -v
```

### Specific Test
```bash
pytest tests/unit/test_models.py::TestUserModel::test_password_hashing
```

## Test Environment

- **Database**: SQLite in-memory (`:memory:`)
- **CSRF**: Disabled for testing
- **Secret Key**: `test-secret-key`
- **Environment**: `TESTING=True` prevents route initialization
- **Isolation**: Each test gets fresh database

## Key Testing Patterns

### 1. Database Testing
```python
def test_example(test_app):
    with test_app.app_context():
        # Test code with database access
        user = User(username='test')
        db.session.add(user)
        db.session.commit()
        assert user.id is not None
```

### 2. API Testing
```python
def test_api(client):
    response = client.post('/api/endpoint', json={'key': 'value'})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'expected_key' in data
```

### 3. Authentication Testing
```python
def test_protected(authenticated_client):
    response = authenticated_client.get('/protected')
    assert response.status_code == 200
```

## CI/CD Integration

The test suite is designed for CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Install dependencies
  run: pip install -r requirements-test.txt

- name: Run tests
  run: pytest --cov=application --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Notes

- Tests use in-memory database (recreated per test)
- External services (Ollama, OpenAI) should be mocked for CI
- CSRF disabled for easier testing
- Sample data automatically created
- Tests are isolated (no cross-test pollution)

## Maintenance

When adding new features:
1. Add unit tests for new models/functions
2. Add integration tests for new endpoints
3. Update fixtures if new sample data needed
4. Update this summary document
5. Run full test suite before committing
