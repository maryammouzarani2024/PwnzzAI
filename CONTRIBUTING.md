# Contributing to PwnzzAI Shop

Thank you for your interest in contributing to PwnzzAI Shop! This project aims to educate developers about LLM security vulnerabilities through practical, hands-on examples. We welcome contributions that enhance the educational value and quality of this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Security Considerations](#security-considerations)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Be kind, professional, and considerate in all interactions.

## Ways to Contribute

You can contribute to PwnzzAI Shop in several ways:

- **Bug Fixes**: Fix issues or improve existing functionality
- **New Vulnerabilities**: Add demonstrations of additional LLM vulnerabilities
- **Documentation**: Improve explanations, tutorials, or documentation
- **Test Coverage**: Add or improve test cases
- **Security Mitigations**: Enhance security soutions in the mitigation strategies sections
- **Model Support**: Add support for additional LLM models
- **UI/UX Improvements**: Enhance the user interface or user experience
- **Code Quality**: Refactor code for better maintainability

## Getting Started

1. **Fork the Repository**: Click the 'Fork' button on the [GitHub repository](https://github.com/maryammouzarani2024/PwnzzAI)

2. **Clone Your Fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/PwnzzAI.git
   cd PwnzzAI
   ```

3. **Add Upstream Remote**:
   ```bash
   git remote add upstream https://github.com/maryammouzarani2024/PwnzzAI.git
   ```

4. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## Development Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager
- (Optional) Docker for containerized development

### Local Setup

1. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   ```

3. **Install System Dependencies** (if needed for testing):
   ```bash
   sudo apt-get update
   sudo apt-get install -y libzbar0
   ```

4. **Configure Environment**:
   - Copy `.flaskenv` if needed
   - Set up any required API keys for OpenAI or Ollama

5. **Run the Application**:
   ```bash
   flask run
   ```

6. **Access the Application**:
   - Visit `http://localhost:5000` in your browser
   - Login credentials: `alice/alice` or `bob/bob`

### Docker Setup

```bash
docker-compose up
```

## Coding Standards

### Python Style Guide

This project uses **Ruff** for linting and code formatting.

- Follow PEP 8 style guidelines
- Current ruff configuration ignores: `E402`, `F401`, `F841`
- Run linting before submitting:
  ```bash
  ruff check . --ignore E402 --ignore F401 --ignore F841
  ```

### Code Organization

- Keep vulnerability demonstrations separate and well-documented
- Use clear, descriptive variable and function names
- Add comments explaining complex security concepts
- Include docstrings for functions and classes

### Commit Messages

Write clear, concise commit messages:

```
Add LLM06 sensitive information disclosure example

- Implement credential extraction demonstration
- Add secure implementation example
- Include mitigation strategies documentation
```

Format:
- Use present tense ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issue numbers when applicable: `Fixes #123`

## Testing

### Running Tests

All contributions should include appropriate tests:

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=application --cov-report=html

# Run specific test file
pytest tests/test_specific.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Use pytest fixtures for common setup
- Test both vulnerable and secure implementations
- Aim for high code coverage on non-vulnerability code

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain or improve existing test coverage

## Submitting Changes

### Pull Request Process

1. **Update Your Branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run Tests and Linting**:
   ```bash
   pytest -v
   ruff check . --ignore E402 --ignore F401 --ignore F841
   ```

3. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**:
   - Go to the [PwnzzAI repository](https://github.com/maryammouzarani2024/PwnzzAI)
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template with:
     - Clear description of changes
     - Related issue numbers
     - Screenshots (if UI changes)
     - Test results

5. **PR Review**:
   - Address reviewer feedback
   - Keep discussions professional and constructive
   - Make requested changes in new commits

### PR Checklist

Before submitting, ensure:

- [ ] Code follows project style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] No sensitive information (API keys, credentials) committed
- [ ] PR targets the `main` branch

## Reporting Issues

### Bug Reports

When reporting bugs, include:

- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Environment details (OS, Python version, browser)
- Screenshots or error messages
- Relevant logs

### Feature Requests

For new features, describe:

- The problem you're trying to solve
- Proposed solution or approach
- How it aligns with the educational goals
- Any relevant OWASP LLM vulnerabilities

### Vulnerability Demonstrations

When proposing new vulnerability demonstrations:

- Reference OWASP LLM documentation
- Explain the educational value
- Outline both vulnerable and secure implementations
- Consider compatibility with OpenAI and Ollama models

## Security Considerations

### Important Reminders

This is an **intentionally vulnerable application** for educational purposes:

- Do not deploy this application in production environments
- Do not use patterns from vulnerable examples in real applications
- Clearly distinguish between vulnerable and secure implementations
- Add warnings and educational context to all vulnerability demonstrations

### Responsible Disclosure

If you discover an **unintentional** security vulnerability (not part of the educational content):

1. Do **NOT** create a public issue
2. Email the maintainers privately
3. Provide details and potential impact
4. Allow time for a fix before public disclosure

## Questions?

If you have questions or need help:

- Open an issue with the `question` label
- Review existing issues and pull requests
- Check the README and documentation

Thank you for contributing to PwnzzAI Shop and helping others learn about LLM security!
