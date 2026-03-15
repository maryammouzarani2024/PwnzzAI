import requests
import time
import json
import os

model_name = ["mistral:7b", "llama3.2:1b"]
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

def start_ollama_service():
    """Start Ollama locally only when using localhost mode."""
    if OLLAMA_BASE_URL != "http://localhost:11434":
        print("Using external Ollama service; skipping local start")
        return True

    try:
        print("Starting Ollama service...")
        # Start ollama serve in the background using shell
        # The '&' at the end runs it in the background
        # Redirect output to /dev/null to avoid blocking
        exit_code = os.system('ollama serve > /dev/null 2>&1 &')

        # Wait a bit for the service to start
        time.sleep(3)

        # Check exit code (0 means command executed successfully)
        if exit_code == 0:
            print("✓ Ollama service started successfully")
            return True
        else:
            print("✗ Ollama service failed to start")
            return False

    except Exception as e:
        print(f"✗ Error starting Ollama service: {e}")
        return False


def check_ollama_running(base_url=OLLAMA_BASE_URL):
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


def ensure_ollama_running(base_url=OLLAMA_BASE_URL, max_retries=3):
    """Ensure Ollama is running, start it if not.
    When base_url points to an external/containerized Ollama (non-localhost),
    skip the local start attempt and just verify connectivity."""
    if base_url is None:
        base_url = OLLAMA_BASE_URL

    # First check if already running
    if check_ollama_running(base_url):
        return True

    # Only attempt local start when Ollama is expected to be on localhost
    is_local = "localhost" in base_url or "127.0.0.1" in base_url
    if is_local:
        print("Attempting to start local Ollama service...")
        started = start_ollama_service()
        if not started:
            return False
    else:
        print(f"External Ollama at {base_url} is not reachable; skipping local start")
        return False

    # Wait and retry checking if it's accessible (local start only)
    for i in range(max_retries):
        time.sleep(2)
        if check_ollama_running(base_url):
            return True
        print(f"Retry {i+1}/{max_retries}...")

    print("✗ Failed to start Ollama service after multiple attempts")
    return False



def is_model_available(model, base_url):
            try:
                response = requests.get(f"{base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json()
                    local_models = [model['name'] for model in models.get('models', [])]
                    print(f"[DEBUG] Looking for '{model}' in available models: {local_models}")
                    is_available = model in local_models
                    print(f"[DEBUG] Model '{model}' available: {is_available}")
                    return is_available
            except Exception as e:
                print(f"Error checking models: {e}")
            return False
        
def check_and_pull_model(model_name, base_url=OLLAMA_BASE_URL):
    """Check if model exists locally, pull if not"""
    for model in model_name:

        print("checking model: ", model)
        # First, check if model exists locally
        if is_model_available(model, base_url):
            print(f"✓ {model} is already available")
            continue

        # Model doesn't exist, pull it
        print(f"✗ {model} not found, pulling...")

        try:
            pull_url = f"{base_url}/api/pull"
            payload = {
                "name": model,
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
                        print(f"✓ {model} pulled successfully!")


                    # Handle errors
                    elif 'error' in data:
                        print(f"✗ Error: {data['error']}")
                        return False



        except requests.exceptions.Timeout:
            print(f"✗ Timeout while pulling {model}")
            return False
        except Exception as e:
            print(f"✗ Error pulling {model}: {e}")
            return False
    return True




def check_and_pull_model_with_progress(model_names, base_url=OLLAMA_BASE_URL):
    
    # Convert single model name to list for uniform handling
    if isinstance(model_names, str):
        model_names = [model_names]

    num_models = len(model_names)

    for idx, model in enumerate(model_names):
        base_progress = (idx / num_models) * 100
        progress_per_model = 100 / num_models

        yield {
            'status': f'Checking model: {model}',
            'progress': base_progress
        }

        if is_model_available(model, base_url):
            yield {
                'status': f'✓ {model} is already available',
                'progress': base_progress + progress_per_model
            }
            continue

        yield {
            'status': f'Pulling {model}...',
            'progress': base_progress + (progress_per_model * 0.1)
        }

        try:
            pull_response = requests.post(
                f"{base_url}/api/pull",
                json={"name": model, "stream": True},
                stream=True,
                timeout=1800
            )

            if pull_response.status_code != 200:
                yield {
                    'status': 'error',
                    'error': f'Failed to pull {model}: {pull_response.text}',
                    'progress': base_progress
                }
                return

            last_progress = 0
            for line in pull_response.iter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except Exception:
                    continue

                status = data.get('status', '')

                if 'total' in data and 'completed' in data and data['total'] > 0:
                    percent = (data['completed'] / data['total']) * 100
                    if percent > last_progress + 5 or percent >= 100:
                        last_progress = percent
                        model_progress = base_progress + (progress_per_model * 0.1) + (percent / 100 * progress_per_model * 0.8)
                        yield {
                            'status': f'Downloading {model}: {percent:.1f}%',
                            'progress': model_progress
                        }
                elif 'verifying' in status.lower():
                    yield {
                        'status': f'Verifying {model} integrity...',
                        'progress': base_progress + (progress_per_model * 0.95)
                    }
                elif 'writing manifest' in status.lower():
                    yield {
                        'status': f'Writing manifest for {model}...',
                        'progress': base_progress + (progress_per_model * 0.98)
                    }
                elif 'success' in status.lower() or data.get('status') == 'success':
                    yield {
                        'status': f'✓ {model} pulled successfully!',
                        'progress': base_progress + progress_per_model
                    }
                elif 'pulling manifest' in status.lower():
                    yield {
                        'status': f'Pulling manifest for {model}...',
                        'progress': base_progress + (progress_per_model * 0.05)
                    }
                elif 'pulling' in status.lower():
                    yield {
                        'status': f'Downloading {model} layers...',
                        'progress': base_progress + (progress_per_model * 0.3)
                    }
                elif 'error' in data:
                    yield {
                        'status': 'error',
                        'error': data['error'],
                        'progress': base_progress
                    }
                    return
        except requests.exceptions.Timeout:
            yield {
                'status': 'error',
                'error': f'Timeout while pulling {model}',
                'progress': base_progress
            }
            return
        except Exception as e:
            yield {
                'status': 'error',
                'error': f'Error pulling {model}: {str(e)}',
                'progress': base_progress
            }
            return

    # All models processed successfully
    yield {
        'status': 'Setup complete! All models are ready.',
        'progress': 100
    }
