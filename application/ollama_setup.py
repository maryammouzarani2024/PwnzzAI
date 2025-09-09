import requests
import subprocess
import sys
import time
import json
model_name="mistral:7b"

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
