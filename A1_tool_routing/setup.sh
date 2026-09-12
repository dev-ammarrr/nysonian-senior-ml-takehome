#!/bin/bash
# Setup script for A1 Tool Routing System

echo "Setting up A1 Tool Routing System..."
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Copy .env.example if .env doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys"
else
    echo ".env already exists"
fi

echo
echo "Setup complete! ✓"
echo
echo "Next steps:"
echo "1. Edit .env and add your API keys"
echo "2. Run: python demo.py"
echo "3. Run tests: python tests/test_router.py"
echo
echo "To activate the virtual environment later:"
echo "  source venv/bin/activate"
