#!/bin/bash
# Environment Setup Script for Tourism MLOps Pipeline

echo "========================================"
echo "Environment Variable Setup"
echo "========================================"
echo ""
echo "Step 1: Create .env file"
echo "------------------------"
cat > .env << 'ENVEOF'
# Hugging Face Token
HF_TOKEN=your_new_token_here
ENVEOF

echo "✓ .env file created"
echo ""
echo "Step 2: Add to your shell profile (optional for persistence)"
echo "-----------------------------------------------------------"
echo "Add this line to your ~/.zshrc or ~/.bash_profile:"
echo ""
echo "export HF_TOKEN='your_new_token_here'"
echo ""
echo "Step 3: Load environment variables"
echo "-----------------------------------"
echo "Run: source .env"
echo ""
echo "Step 4: Verify"
echo "--------------"
echo "Run: echo \$HF_TOKEN"
echo ""
echo "========================================"
echo "IMPORTANT: Get your NEW token from:"
echo "https://huggingface.co/settings/tokens"
echo "========================================"
