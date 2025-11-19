import requests
import time
import json
import os
model_name=["mistral:7b", "llama3.2:1b"]

def start_ollama_service():
    """Start Ollama service in the background using os.system()"""
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
    started = start_ollama_service()

    if not started:
        return False

    # Wait and retry checking if it's accessible
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
        
def check_and_pull_model(model_name, base_url="http://localhost:11434"):
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


# def check_and_pull_model_with_progress(model_name, base_url="http://localhost:11434"):
#     """
#     Generator function that yields progress updates while checking and pulling models.
#     Yields dict with 'status', 'progress', and optionally 'error' keys.
#     """
#     num_models = len(model_name)

#     for idx, model in enumerate(model_name):
#         print("checking: ",model)
#         # Base progress per model
#         base_progress = (idx / num_models) * 100
#         progress_per_model = 100 / num_models

#         yield {
#             'status': f'Checking model: {model}',
#             'progress': base_progress + (progress_per_model * 0.1)
#         }

#         # Check if model exists locally
#         if is_model_available(model, base_url):
#             yield {
#                 'status': f'✓ {model} is already available',
#                 'progress': base_progress + progress_per_model
#             }
#             continue

#         # Model doesn't exist, pull it
#         yield {
#             'status': f'Pulling {model}...',
#             'progress': base_progress + (progress_per_model * 0.2)
#         }

#         try:
#             pull_url = f"{base_url}/api/pull"
#             payload = {
#                 "name": model,  # Ollama API uses "name" not "model" for pull endpoint
#                 "stream": True
#             }

#             response = requests.post(pull_url, json=payload, stream=True, timeout=1800)

#             if response.status_code != 200:
#                 yield {
#                     'status': 'error',
#                     'error': f'Failed to start pull: {response.text}',
#                     'progress': base_progress
#                 }
#                 return

#             # Process streaming response
#             pull_completed = False
#             for line in response.iter_lines():
#                 if line:
#                     print("RAW LINE:", line)
#                     data = json.loads(line)
#                     status = data.get('status', '')

#                     # Debug: print the raw response
#                     print(f"[DEBUG] Ollama response: {data}")

#                     # Show download progress
#                     if 'downloading' in status.lower() or 'pulling' in status.lower():
#                         if 'total' in data and 'completed' in data:
#                             percent = (data['completed'] / data['total']) * 100
#                             # Scale progress to current model's portion
#                             model_progress = base_progress + (progress_per_model * 0.2) + (percent / 100 * progress_per_model * 0.8)
#                             yield {
#                                 'status': f'Downloading {model}: {percent:.1f}%',
#                                 'progress': model_progress
#                             }
#                         else:
#                             yield {
#                                 'status': f'{status}',
#                                 'progress': base_progress + (progress_per_model * 0.5)
#                             }

#                     # Check if complete
#                     elif 'success' in status.lower() or data.get('status') == 'success':
#                         pull_completed = True
#                         yield {
#                             'status': f'✓ {model} pulled successfully!',
#                             'progress': base_progress + progress_per_model
#                         }
#                         break  # Exit loop after success

#                     # Handle errors
#                     elif 'error' in data:
#                         yield {
#                             'status': 'error',
#                             'error': f"Error: {data['error']}",
#                             'progress': base_progress
#                         }
#                         return

#             # Ensure the model was actually pulled
#             if not pull_completed:
#                 if is_model_available(model, base_url):
#                     print("[DEBUG] Warning: Ollama ended stream silently, but model appears installed.")
#                     pull_completed = True
#                 else:
#                     print("[DEBUG] ERROR: Stream ended early, model NOT installed.")
#                     yield { "status": "error", "error": f"Model {model} pull failed: stream ended early." }
#                     return

#         except requests.exceptions.Timeout:
#             yield {
#                 'status': 'error',
#                 'error': f'Timeout while pulling {model}',
#                 'progress': base_progress
#             }
#             return
#         except Exception as e:
#             yield {
#                 'status': 'error',
#                 'error': f'Error pulling {model}: {e}',
#                 'progress': base_progress
#             }
#             return

#     # All models processed successfully
#     yield {
#         'status': 'Setup complete! All models are ready.',
#         'progress': 100
#     }


import subprocess
import re

def check_and_pull_model_with_progress(model_names, base_url="http://localhost:11434"):
    """
    Generator function that pulls models via the Ollama CLI and yields progress updates.
    Handles both a single model name (string) or a list of model names.
    """
    # Convert single model name to list for uniform handling
    if isinstance(model_names, str):
        model_names = [model_names]

    num_models = len(model_names)

    for idx, model in enumerate(model_names):
        # Base progress per model
        base_progress = (idx / num_models) * 100
        progress_per_model = 100 / num_models

        yield {
            'status': f'Checking model: {model}',
            'progress': base_progress
        }

        # Check if model exists locally
        if is_model_available(model, base_url):
            yield {
                'status': f'✓ {model} is already available',
                'progress': base_progress + progress_per_model
            }
            continue

        # Model doesn't exist, pull it
        yield {
            'status': f'Pulling {model}...',
            'progress': base_progress + (progress_per_model * 0.1)
        }

        cmd = ["ollama", "pull", model]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        last_progress = 0
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue

            # Debug
            print(f"[DEBUG] CLI output: {line}")

               # Try multiple patterns to match Ollama's output
                  # Pattern 1: "pulling 74701a8c35f6... 100%"
            m = re.search(r'pulling\s+\w+.*?\s+(\d+)%', line)
            if not m:
                      # Pattern 2: "▕████████████▏ 100%"
               m = re.search(r'(\d+)%', line)
            if m:
                 # Only yield if progress has increased significantly (avoid spam)
                percent = float(m.group(1))
                if percent > last_progress + 5 or percent >= 100:
                             last_progress = percent
                            # Scale progress to current model's portion
                             model_progress = base_progress + (progress_per_model * 0.1) + (percent / 100 * 
                                progress_per_model * 0.8)
                             yield {
                                 'status': f'Downloading {model}: {percent:.1f}%',
                                 'progress': model_progress
                            }

            elif "verifying" in line.lower() and "digest" in line.lower():
                yield {
                    'status': f'Verifying {model} integrity...',
                    'progress': base_progress + (progress_per_model * 0.95)
                }

            elif "writing manifest" in line.lower():
                yield {
                    'status': f'Writing manifest for {model}...',
                    'progress': base_progress + (progress_per_model * 0.98)
                }

            elif "success" in line.lower():
                yield {
                    'status': f'✓ {model} pulled successfully!',
                    'progress': base_progress + progress_per_model
                }
                
            elif "pulling manifest" in line.lower():
                    yield {
                            'status': f'Pulling manifest for {model}...',
                             'progress': base_progress + (progress_per_model * 0.05)
                    }
         
            elif "pulling" in line.lower() and "%" not in line:
                        # Generic pulling message without percentage
                    yield {
                             'status': f'Downloading {model} layers...',
                             'progress': base_progress + (progress_per_model * 0.3)
                    }    

        process.wait()
        if process.returncode != 0:
            yield {
                'status': 'error',
                'error': f'CLI pull failed for {model} with return code {process.returncode}',
                'progress': base_progress
            }
            return

    # All models processed successfully
    yield {
        'status': 'Setup complete! All models are ready.',
        'progress': 100
    }
