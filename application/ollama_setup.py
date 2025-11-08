import requests
import subprocess
import time
import json
import os
model_name="mistral:7b"

def start_ollama_service():
    """Start Ollama service in the background"""
    try:
        print("Starting Ollama service...")
        # Start ollama serve in the background
        process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setpgrp  # Create new process group
        )

        # Wait a bit for the service to start
        time.sleep(3)

        # Check if process is still running
        if process.poll() is None:
            print("✓ Ollama service started successfully")
            return process
        else:
            print("✗ Ollama service failed to start")
            return None

    except FileNotFoundError:
        print("✗ Ollama command not found. Is Ollama installed?")
        return None
    except Exception as e:
        print(f"✗ Error starting Ollama service: {e}")
        return None


def check_ollama_running(base_url="http://localhost:11434"):
    """Check if Ollama service is running"""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✓ Ollama is running and accessible")
            return True
    except requests.exceptions.ConnectionError:
        print("✗ Ollama is not running or not accessible")
    except requests.exceptions.Timeout:
        print("✗ Ollama is not responding (timeout)")
    except Exception as e:
        print(f"✗ Error connecting to Ollama: {e}")

    return False


def ensure_ollama_running(base_url="http://localhost:11434", max_retries=3):
    """Ensure Ollama is running, start it if not"""
    # First check if already running
    if check_ollama_running(base_url):
        return True

    # Try to start Ollama
    print("Attempting to start Ollama...")
    process = start_ollama_service()

    if process is None:
        return False

    # Wait and retry checking if it's accessible
    for i in range(max_retries):
        time.sleep(2)
        if check_ollama_running(base_url):
            return True
        print(f"Retry {i+1}/{max_retries}...")

    print("✗ Failed to start Ollama service after multiple attempts")
    return False




def check_and_pull_model(model_name, base_url="http://localhost:11434"):
    """Check if model exists locally, pull if not"""
    
    # First, check if model exists locally
    def is_model_available():
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code == 200:
                models = response.json()
                local_models = [model['name'] for model in models.get('models', [])]
                return model_name in local_models
        except Exception as e:
            print(f"Error checking models: {e}")
        return False
    
    # If model exists, return True
    if is_model_available():
        print(f"✓ {model_name} is already available")
        return True
    
    # Model doesn't exist, pull it
    print(f"✗ {model_name} not found, pulling...")
    
    try:
        pull_url = f"{base_url}/api/pull"
        payload = {
            "name": model_name,
            "stream": True  # Show progress
        }
        
        response = requests.post(pull_url, json=payload, stream=True, timeout=1800)
        
        if response.status_code != 200:
            print(f"Failed to start pull: {response.text}")
            return False
        
        # Process streaming response
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get('status', '')
                
                # Show download progress
                if 'downloading' in status.lower() or 'pulling' in status.lower():
                    if 'total' in data and 'completed' in data:
                        percent = (data['completed'] / data['total']) * 100
                        print(f"  Progress: {percent:.1f}%")
                    else:
                        print(f"  {status}")
                
                # Check if complete
                elif 'success' in status.lower() or data.get('status') == 'success':
                    print(f"✓ {model_name} pulled successfully!")
                    return True
                
                # Handle errors
                elif 'error' in data:
                    print(f"✗ Error: {data['error']}")
                    return False
        
        return True
        
    except requests.exceptions.Timeout:
        print(f"✗ Timeout while pulling {model_name}")
        return False
    except Exception as e:
        print(f"✗ Error pulling {model_name}: {e}")
        return False
