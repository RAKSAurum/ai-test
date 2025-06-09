#!/bin/bash

# AI Developer Challenge Application Startup Script
# Handles dependency management, environment setup, and application launch
# with comprehensive error handling and validation

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Configuration variables
readonly SCRIPT_NAME="$(basename "$0")"
readonly PROJECT_NAME="AI Developer Challenge Application"
readonly PROJECT_VERSION="core v1.0"
readonly REQUIRED_PYTHON_MIN="3.9"
readonly REQUIRED_PYTHON_MAX="4.0"
readonly APPLICATION_PORT="8888"
readonly APPLICATION_ENTRY="ignite.py"

# Color codes for enhanced output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions for consistent output
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}" >&2
}

# Enhanced startup banner with project information
print_startup_banner() {
    echo "🚀 Starting ${PROJECT_NAME}..."
    echo "📋 Project: ${PROJECT_VERSION} - AI Skills Demonstration"
    echo "🔧 Script: ${SCRIPT_NAME}"
    echo "📅 Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "----------------------------------------"
}

# Comprehensive Poetry installation check with installation guidance
check_poetry_installation() {
    log_info "Checking Poetry installation..."
    
    if ! command -v poetry &> /dev/null; then
        log_error "Poetry not found. Poetry is required for dependency management."
        echo ""
        echo "📦 Install Poetry using one of these methods:"
        echo "   Method 1 (Recommended): curl -sSL https://install.python-poetry.org | python3 -"
        echo "   Method 2 (pip): pip install --user poetry"
        echo "   Method 3 (Homebrew): brew install poetry"
        echo ""
        echo "After installation, restart your terminal and run this script again."
        exit 1
    fi
    
    local poetry_version
    poetry_version=$(poetry --version 2>/dev/null | cut -d' ' -f3 || echo "unknown")
    log_success "Poetry found (version: ${poetry_version})"
}

# Enhanced Python version validation with detailed compatibility checking
validate_python_version() {
    log_info "Validating Python version compatibility..."
    
    # Check if python3 is available
    if ! command -v python3 &> /dev/null; then
        log_error "python3 command not found. Please install Python 3.9 or higher."
        exit 1
    fi
    
    # Get detailed Python version information
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    local python_major_minor
    python_major_minor=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    
    echo "🐍 Python version: ${python_version}"
    
    # Validate version range (>=3.9,<4.0)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) and sys.version_info < (4,0) else 1)" 2>/dev/null; then
        log_error "Python version ${python_version} is not compatible."
        echo ""
        echo "📋 Requirements:"
        echo "   - Minimum: Python ${REQUIRED_PYTHON_MIN}"
        echo "   - Maximum: Python ${REQUIRED_PYTHON_MAX} (exclusive)"
        echo "   - Current: Python ${python_version}"
        echo ""
        echo "Please install a compatible Python version and try again."
        exit 1
    fi
    
    log_success "Python version ${python_version} is compatible"
}

# Robust dependency installation with error handling and progress feedback
install_dependencies() {
    log_info "Installing project dependencies with Poetry..."
    
    # Check if pyproject.toml exists
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found. Ensure you're in the correct project directory."
        exit 1
    fi
    
    # Install dependencies excluding development packages for production efficiency
    if ! poetry install --without dev --no-interaction; then
        log_error "Failed to install dependencies. Check your pyproject.toml configuration."
        echo ""
        echo "🔧 Troubleshooting steps:"
        echo "   1. Verify pyproject.toml syntax"
        echo "   2. Check network connectivity"
        echo "   3. Try: poetry lock --no-update"
        echo "   4. Try: poetry install --verbose"
        exit 1
    fi
    
    log_success "Dependencies installed successfully"
}

# Comprehensive OpenFabric SDK verification with detailed error reporting
verify_openfabric_installation() {
    log_info "Verifying openfabric-pysdk installation..."
    
    # Test import with detailed error handling
    if ! poetry run python3 -c "
import sys
try:
    import openfabric_pysdk
    print('✅ openfabric-pysdk imported successfully')
    print(f'📦 Version: {getattr(openfabric_pysdk, \"__version__\", \"unknown\")}')
except ImportError as e:
    print(f'❌ Import failed: {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
        log_error "Failed to import openfabric-pysdk"
        echo ""
        echo "🔧 Troubleshooting steps:"
        echo "   1. Verify openfabric-pysdk is in pyproject.toml dependencies"
        echo "   2. Try: poetry add openfabric-pysdk"
        echo "   3. Check for version conflicts: poetry show --tree"
        echo "   4. Try: poetry install --verbose"
        exit 1
    fi
}

# Enhanced environment variable loading with validation and security
load_environment_variables() {
    if [[ -f ".env" ]]; then
        log_info "Loading environment variables from .env file..."
        
        # Validate .env file format and load safely
        if grep -q "^[A-Za-z_][A-Za-z0-9_]*=" .env 2>/dev/null; then
            # Use a safer method to load environment variables
            set -a  # Mark variables for export
            # shellcheck source=/dev/null
            source .env 2>/dev/null || {
                log_warning "Some environment variables failed to load"
            }
            set +a  # Unmark variables for export
            log_success "Environment variables loaded"
        else
            log_warning ".env file exists but appears to be empty or malformed"
        fi
    else
        log_info "No .env file found (optional)"
    fi
}

# Pre-flight checks to ensure application readiness
perform_preflight_checks() {
    log_info "Performing pre-flight checks..."
    
    # Check if main application file exists
    if [[ ! -f "${APPLICATION_ENTRY}" ]]; then
        log_error "Application entry point '${APPLICATION_ENTRY}' not found"
        exit 1
    fi
    
    # Check if port is available (optional check)
    if command -v lsof &> /dev/null; then
        if lsof -i ":${APPLICATION_PORT}" &> /dev/null; then
            log_warning "Port ${APPLICATION_PORT} appears to be in use"
            echo "   The application may fail to start or use a different port"
        fi
    fi
    
    log_success "Pre-flight checks completed"
}

# Enhanced application startup with comprehensive information display
start_application() {
    echo ""
    echo "🎯 Starting application..."
    echo "🌐 Server will be available at: http://localhost:${APPLICATION_PORT}"
    echo "📊 Swagger UI: http://localhost:${APPLICATION_PORT}/swagger-ui/#/"
    echo "📁 Output directory: $(pwd)/outputs"
    echo "📝 Log directory: $(pwd)/logs"
    echo "🧠 Memory database: $(pwd)/memory"
    echo ""
    echo "💡 Usage tips:"
    echo "   - Monitor application logs in the logs/ directory"
    echo "   - Generated files will be saved in outputs/ directory"
    echo "   - Memory system stores all generations for recall"
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo "========================================"
    echo ""
    
    # Set up signal handling for graceful shutdown
    trap 'echo ""; log_info "Shutting down application..."; exit 0' INT TERM
    
    # Start the application with error handling
    if ! poetry run python3 "${APPLICATION_ENTRY}"; then
        log_error "Application failed to start or crashed"
        echo ""
        echo "🔧 Troubleshooting:"
        echo "   - Check logs/ directory for error details"
        echo "   - Verify all dependencies are installed"
        echo "   - Ensure port ${APPLICATION_PORT} is available"
        echo "   - Try running: poetry run python3 ${APPLICATION_ENTRY} --debug"
        exit 1
    fi
}

# Graceful shutdown handler
cleanup_and_exit() {
    echo ""
    log_success "Application stopped successfully!"
    log_info "Session ended: $(date '+%Y-%m-%d %H:%M:%S')"
    exit 0
}

# Main execution flow with comprehensive error handling
main() {
    # Set up error handling
    trap 'log_error "Script failed at line $LINENO. Exit code: $?"' ERR
    trap cleanup_and_exit EXIT
    
    # Execute startup sequence
    print_startup_banner
    check_poetry_installation
    validate_python_version
    install_dependencies
    verify_openfabric_installation
    load_environment_variables
    perform_preflight_checks
    start_application
}

# Execute main function
main "$@"