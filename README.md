# 🍕 Welcome to PwnzzAI(/pəʊnzɑː/) Shop – The Vulnerable Pizza Shop! 💻

<img src="application/static/img/index.png" alt="PwnzzAI Shop" width="200">

At PwnzzAI Shop, every slice serves up a helping of **holistic AI security instruction**. This hands-on learning platform prepares professionals to understand the protection of **AI technologies broadly** delivered through an engaging, practical pizza shop scenario.

Just as an incorrect ingredient can spoil a pizza, a flawed prompt or inadequate architecture can leave AI technologies vulnerable to severe threats, including information breaches, intellectual property theft, or illegitimate access.

Here, you'll explore **practical examples** of how vulnerabilities are created, exploited, and mitigated. You need to login as alice/alice or bob/bob for some pages. Grab a slice, dig in, and discover how delicious learning about AI security can be.

**Table of Content**
- [About](#about)
  - [Scope and Learning Goals](#scope-and-learning-goals)
- [Setup Instructions](#setup-instructions)
  - [Option 1: Using Docker (Recommended)](#option-1-using-docker-recommended)
  - [Option 2: Local Setup](#option-2-local-setup)
- [Features](#features)
- [AI Security Coverage](#ai-security-coverage)
  - [Learning Framework](#learning-framework)
  - [Implemented Vulnerabilities](#implemented-vulnerabilities)
  - [Model Support](#model-support)

## About

PwnzzAI Shop represents a **hands-on learning platform** purposefully built for instruction on AI technology protection. This intentionally insecure Flask-based web application showcases an extensive array of AI security weaknesses via an immersive pizza shop experience.

### Scope and Learning Goals

PwnzzAI is designed as a comprehensive, evolving learning platform aligned with the **[AI Exchange risk taxonomy](https://owaspai.org/docs/ai_security_overview/)** Its scope expands progressively in step with OWASP AI Exchange security analysis frameworks and defensive guidance, ensuring long-term relevance as AI security practices mature.
The project currently incorporates the OWASP Top 10 for LLMs, with an architecture intentionally mapped to AI Exchange risk classifications. Over time, PwnzzAI aims to mature into a structured learning ecosystem that supports:

- AI protection and defense training, including self-directed learning paths

- Hands-on security training programs with practical, scenario-driven exercises

- Team-based learning initiatives for security engineers and practitioners

- End-to-end AI security education, spanning design, development, deployment, and operations
  
## Setup Instructions

### Option 1: Using Docker (Recommended)

The easiest way to get started is using our pre-built Docker image:

1. Pull the Docker image:
   ```bash
   docker pull ghcr.io/maryammouzarani2024/pwnzzai:latest
   ```

2. Run the container:
   ```bash
   docker run -p 5000:5000 ghcr.io/maryammouzarani2024/pwnzzai:latest
   ```

3. Visit `http://localhost:5000` in your browser to see the application. Start from the Basic page and setup your lab.

The image already includes Ollama so you only need to pull the required models as described in the lab setup section of the Basics page. 

### Option 2: Local Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install  the dependencies:
   
   (If you do not want to use Ollam, you can remove or comment the line "curl -fsSL https://ollama.com/install.sh | sh" from install.sh)
   
   
   ```bash
   ./install.sh
   ```

3. Run the application:
   ```bash
   flask run
   ```

4. Visit `http://localhost:5000` in your browser to see the application. Start from the Basic page and setup your lab. 


If you need more details, watch <a href="https://www.youtube.com/watch?v=Pv3PP6xbS3A&t=26s"> this walkthrough </a>, that shows, step-by-step, how to set up the app using either options. 

## Features

- **Pizza Service**: Browse appetizing pizzas, explore customer feedback, contribute comments, and place virtual orders!
- **Extensive Weakness Showcases**: Active demonstrations spanning OWASP LLM Top 10 and AI Exchange threat categories including model input threats, development-time threats, runtime security threats, and system-level AI risks.
- **Learning-Focused Design**: Discover exploitation methods and defensive approaches via transparent descriptions, crafted for organized instruction programs and aligned with [AI Exchange threat taxonomy](https://owaspai.org/docs/ai_security_overview/).
- **Multi-Provider Implementation**: Every demonstration operates with both commercial OpenAI and complimentary free Ollama frameworks, ensuring accessibility for all participants.
- **Standards-Based Content**: Educational materials synchronized with [OWASP AI Exchange](https://owaspai.org/), which serves as the foundational basis for emerging AI security standards such as ISO/IEC 27090 and prEN 18282.


## AI Security Coverage

This platform showcases an extensive spectrum of AI protection challenges, aligned with the **[OWASP AI Exchange comprehensive threat taxonomy](https://owaspai.org/docs/ai_security_overview/)**. Current implementation covers the OWASP Top 10 for LLM Applications and core AI Exchange threat categories, with an extensible architecture built for future AI risk coverage.

### Learning Framework

Every weakness demonstrated in PwnzzAI features:

1. **Active demonstrations** illustrating the vulnerability's mechanics
2. **Exploitation scenarios** providing interactive attack exercises
3. **Hardened alternatives** detailing defensive approaches

### Implemented Vulnerabilities:
According to <a href="https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/" > OWASP Top 10 for LLM Applications 2025</a> and [OWASP AI Exchange threats](https://owaspai.org/docs/ai_security_overview/):

1. **LLM01: Prompt Injection** ([AI Exchange: Direct Prompt Injection](https://owaspai.org/goto/directpromptinjection/), [Indirect Prompt Injection](https://owaspai.org/goto/indirectpromptinjection/))
   - **Direct Prompt Injection**: Bypass system instructions through crafted user inputs to manipulate model behavior.
   - **Indirect Prompt Injection**: Exploit external data sources to inject malicious instructions and manipulate model responses.
   - *AI Exchange Context*: Model input threats where attackers craft instructions to deceive the model.

2.  **LLM02: Sensitive Information Disclosure** ([AI Exchange: Data Disclosure](https://owaspai.org/goto/disclosureuse/), [Training Data Leaks](https://owaspai.org/goto/devdataleak/))
   - Extraction of training data, system information, and credentials through model outputs.
   - *AI Exchange Context*: Training data confidentiality threats including model inversion, membership inference, and sensitive data output from model.

3.  **LLM03: Supply Chain Vulnerabilities** ([AI Exchange: Supply Chain Model Poisoning](https://owaspai.org/goto/supplymodelpoison/))
   - Third-party model and plugin security risks from compromised suppliers.
   - *AI Exchange Context*: Development-time supply chain threats including poisoned pre-trained models, corrupted data sources, and compromised model hosting.

4. **LLM04: Data and Model Poisoning** ([AI Exchange: Data Poisoning](https://owaspai.org/goto/datapoison/), [Model Poisoning](https://owaspai.org/goto/modelpoison/))
   - Demonstrate how malicious training data affects model responses and behavior.
   - *AI Exchange Context*: Development-time threats where training data manipulation or direct model parameter tampering leads to unwanted model behavior.

5. **LLM05: Improper Output Handling** ([AI Exchange: Insecure Output](https://owaspai.org/goto/insecureoutput/))
   - Unvalidated LLM outputs leading to XSS and other injection attacks in downstream systems.
   - *AI Exchange Context*: Runtime application security threats where model output contains malicious code or injection attacks.

6. **LLM06: Excessive Agency** ([AI Exchange: Least Model Privilege](https://owaspai.org/goto/leastmodelprivilege/), [Oversight](https://owaspai.org/goto/oversight/))
   - Over-privileged LLM operations and autonomous actions without appropriate constraints.
   - *AI Exchange Context*: Impact limitation controls to restrict unwanted model behavior through privilege management and human oversight, particularly critical for agentic AI systems.

7. **LLM07: System Prompt Leakage** ([AI Exchange: Sensitive Information Disclosure](https://owaspai.org/goto/disclosureuse/))
   - Extraction of secret information embedded in system prompts through manipulation techniques.
   - *AI Exchange Context*: Model use threats where attackers extract confidential instructions or configuration data through crafted inputs.

8. **LLM08: Vector and Embedding Weakness** ([AI Exchange: Data Confidentiality](https://owaspai.org/goto/devdataleak/))
   - Unauthorized access to embeddings containing sensitive information in RAG systems or vector databases.
   - *AI Exchange Context*: Augmentation data threats where retrieval repositories or embedding stores leak confidential information.

9. **LLM09: Misinformation** ([AI Exchange: Model Alignment](https://owaspai.org/goto/modelalignment/), [Continuous Validation](https://owaspai.org/goto/continuousvalidation/))
   - Critical decision-making without human oversight leading to harmful misinformation or hallucinations.
   - *AI Exchange Context*: Unwanted model behavior risks requiring oversight controls, explainability, and continuous monitoring to ensure accuracy and safety.

10. **LLM10: Unbounded Consumption** ([AI Exchange: AI Resource Exhaustion](https://owaspai.org/goto/airesourceexhaustion/))
   - Resource exhaustion attacks and rate limiting bypass techniques causing denial of service.
   - *AI Exchange Context*: Model availability threats through resource depletion via excessive or crafted inputs.

and, 

11. **Model Theft** (in earlier OWASP Top 10 for LLM versions) ([AI Exchange: Model Theft Through Use](https://owaspai.org/goto/modeltheftuse/), [Runtime Model Theft](https://owaspai.org/goto/runtimemodeltheft/))
    - Model extraction and intellectual property theft through API abuse and input-output harvesting.
    - *AI Exchange Context*: Model confidentiality threats including model parameter extraction, model replication, and intellectual property theft through various attack vectors.


### Model Support:
- **OpenAI Models**: GPT-3.5/GPT-4 demonstrations via OpenAI API.
- **Ollama Models**: Free models, such as Mistral 7B and LLaMA 3.2 1B, are accessible through Ollama.


**⚠️ Educational Purpose Only**: This application contains intentional security vulnerabilities. Do not use in production environments.
