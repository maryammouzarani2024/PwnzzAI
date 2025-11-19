#!/bin/bash
sudo apt update
sudo apt install -y libzbar0
curl -fsSL https://ollama.com/install.sh | sh
pip install -r requirements.txt