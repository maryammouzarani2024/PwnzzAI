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
