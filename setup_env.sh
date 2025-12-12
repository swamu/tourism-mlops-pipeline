#!/bin/bash
# Environment Setup Script for Tourism MLOps Pipeline

echo "========================================"
echo "Environment Variable Setup"
echo "========================================"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file template..."
    cat > .env << 'ENVEOF'
# Hugging Face Token
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN=your_new_token_here
ENVEOF
    echo "✓ .env file created"
    echo ""
    echo "Please edit .env file and add your HF token, then run:"
    echo "  source setup_env.sh"
else
    # Load environment variables from .env file
    export $(cat .env | grep -v '^#' | xargs)
    
    if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" = "your_new_token_here" ]; then
        echo "WARNING: HF_TOKEN not set properly!"
        echo ""
        echo "Steps to fix:"
        echo "1. Get your token from: https://huggingface.co/settings/tokens"
        echo "2. Edit .env file: nano .env"
        echo "3. Replace 'your_new_token_here' with your actual token"
        echo "4. Run: source setup_env.sh"
    else
        echo "✓ HF_TOKEN loaded successfully"
        echo "✓ Token starts with: ${HF_TOKEN:0:8}..."
        echo ""
        echo "Environment is ready!"
    fi
fi

echo ""
echo "========================================"
