# DVLLM - Vulnerable Pizza Shop Demo

This is an educational Flask web application demonstrating various LLM security vulnerabilities based on OWASP Top 10 for LLMs.

## Setup Instructions

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the dependencies:
   ```
   pip install flask flask-sqlalchemy
   ```

3. Run the application:
   ```
   flask run
   ```

4. Visit `http://localhost:5000` in your browser to see the application

## Features

- View a list of pizzas
- View details about individual pizzas
- Read and add customer reviews
- Interact with a simple pizza assistant (containing LLM vulnerabilities for educational purposes)

## Vulnerabilities

This application intentionally contains various LLM vulnerabilities for educational purposes.
Each vulnerability will be introduced in different parts of the application to demonstrate:

1. How the vulnerability works
2. How it can be exploited
3. How to properly mitigate it

The following vulnerabilities are demonstrated:

1. **Model Theft** - Shows how attackers can extract model weights through repeated API queries
2. **Training Data Poisoning** - Demonstrates how malicious data can manipulate model behavior
3. **LLM DoS Simulation** - Simulates a denial of service attack with synthetic degradation
4. **LLM Real DoS** - Demonstrates a real DoS attack against OpenAI API alongside secure rate-limited implementation

Note: This is for educational purposes only. Do not use this code in production environments.