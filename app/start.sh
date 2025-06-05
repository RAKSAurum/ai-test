#!/bin/bash

set -e

echo "🚀 Starting AI Developer Challenge Application..."
echo "📋 Project: core v1.0 - AI Skills Demonstration"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check Python version compatibility (>=3.9,<4.0)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $PYTHON_VERSION"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) and sys.version_info < (4,0) else 1)"; then
    echo "❌ Python version must be >=3.9,<4.0"
    exit 1
fi

# Install dependencies using modern Poetry syntax
echo "📦 Installing dependencies with Poetry..."
poetry install --without dev

# Alternative options you can use:
# poetry install --only main           # Install only main dependencies
# poetry install                       # Install all dependencies including dev
# poetry install --with gui            # Include optional GUI dependencies

# Create necessary directories for the AI pipeline
echo "📁 Creating project directories..."
mkdir -p outputs memory logs models

# Check if openfabric-pysdk is properly installed
echo "🔍 Verifying openfabric-pysdk installation..."
poetry run python3 -c "import openfabric_pysdk; print('✅ openfabric-pysdk imported successfully')" || {
    echo "❌ Failed to import openfabric-pysdk"
    exit 1
}

# Set environment variables if .env file exists
if [ -f ".env" ]; then
    echo "🔧 Loading environment variables..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Start the application using Poetry
echo "🎯 Starting application on port 8888..."
echo "🌐 Access Swagger UI at: http://localhost:8888/swagger-ui/#/"
echo "📊 Monitor logs in the logs/ directory"
echo ""
echo "Press Ctrl+C to stop the application"
echo "----------------------------------------"

# Run the application through Poetry's virtual environment
poetry run python3 ignite.py

echo "✅ Application stopped successfully!"