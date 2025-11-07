# 🍕 Welcome to PwnzzAI {###(/pəʊnzɑː/)###} Shop – The Vulnerable Pizza Shop! 💻

<img src="application/static/img/index.png" alt="PwnzzAI Shop" width="200">

At PwnzzAI Shop, every slice comes with a side of **AI security lessons**. This educational web app serves up the **OWASP Top 10 LLM Vulnerabilities** in a fun, hands-on environment.

Just as the wrong ingredient can ruin a pizza, the wrong prompt or poor design can expose AI systems to serious risks, like data leaks, model theft, or unauthorized access.

Here, you'll explore **practical examples** of how vulnerabilities are created, exploited, and mitigated. You need to login as alice/alice or bob/bob for some pages. Grab a slice, dig in, and discover how delicious learning about AI security can be.

## About

PwnzzAI Shop is a deliberately vulnerable Large Language Model application. This is an educational Flask web application demonstrating various LLM security vulnerabilities based on OWASP Top 10 for LLMs through an interactive pizza shop. 

## Setup Instructions

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   flask run
   ```

4. Visit `http://localhost:5000` in your browser to see the application. Start from the Basic page and setup your lab. 

## Features

- **Pizza Service**: You can browse delicious pizzas, read the customer reviews, add comments and order pizza virtually!
- **Vulnerability Demonstrations**: Live examples of OWASP LLM Top 10 vulnerabilities
- **Educational Interface**:  Learn about attack techniques and mitigation strategies through clear explanations.
- **Dual Model Support**: All examples are implemented using both OpenAI and Free Ollama models to be accessible and useful for everyone.

## OWASP LLM Top 10 Vulnerabilities

This application demonstrates the complete OWASP Top 10 for Large Language Model Applications. Each vulnerability includes:

1. **Live demonstrations** showing how the vulnerability works
2. **Exploitation examples** An interactive attack demonstration
3. **Secure implementations** Explaining mitigation strategies

### Implemented Vulnerabilities:

1. **LLM01: Prompt Injection**
   - **Direct Prompt Injection**: Bypass system instructions through crafted user inputs
   - **Indirect Prompt Injection**: Exploit external data sources to manipulate model behavior

2. **LLM02: Insecure Output Handling**
   - Unvalidated LLM outputs leading to XSS and other injection attacks

3. **LLM03: Training Data Poisoning**
   - Demonstrate how malicious training data affects model responses

4. **LLM04: Model Denial of Service**
   - Resource exhaustion attacks and rate limiting bypass techniques

5. **LLM05: Supply Chain Vulnerabilities**
   - Third-party model and plugin security risks

6. **LLM06: Sensitive Information Disclosure**
   - Extraction of training data, system information, and credentials

7. **LLM07: Insecure Plugin Design**
   - Unsafe plugin architectures and implementations

8. **LLM08: Excessive Agency**
   - Over-privileged LLM operations and autonomous actions

9. **LLM09: Overreliance**
   - Critical decision-making without human oversight leading to misinformation

10. **LLM10: Model Theft**
    - Model extraction and intellectual property theft through API abuse

### Model Support:
- **OpenAI Models**: GPT-3.5/GPT-4 demonstrations via API
- **Ollama Models**: Free models, such as Mistral 7B and LLaMA 3.2 1B, are accessible through Ollama.

✅ Improvements: 
**⚠️ Educational Purpose Only**: This application contains intentional security vulnerabilities. Do not use in production environments.