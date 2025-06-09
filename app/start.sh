#!/bin/bash

# AI Developer Challenge Application Startup Script
# Handles dependency management, environment setup, and application launch
# with comprehensive error handling and validation

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Configuration variables
readonly SCRIPT_NAME="$(basename "$0")"
readonly PROJECT_NAME="AI 3D Generator with Intelligent Memory System"
readonly PROJECT_VERSION="core v1.0"
readonly REQUIRED_PYTHON_MIN="3.9"
readonly REQUIRED_PYTHON_MAX="3.11"
readonly API_PORT="8888"
readonly GUI_PORT="7860"
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

# Check and request elevated permissions
check_and_elevate_permissions() {
    log_info "Checking file system permissions..."
    
    # Check if running as root
    if [[ $EUID -eq 0 ]]; then
        log_success "Running with elevated permissions"
        return 0
    fi
    
    # Test write access to system directories
    if [[ ! -w "/usr/local" ]] || [[ ! -w "/opt" ]]; then
        log_warning "Limited file system access detected"
        echo ""
        echo "🔐 This script requires elevated permissions for full file system access."
        echo "   Re-running with sudo while preserving environment..."
        echo ""
        
        # Get the absolute path of the current script
        SCRIPT_PATH="$(readlink -f "$0")"
        
        # Re-run script with sudo while preserving environment
        exec sudo -E bash "$SCRIPT_PATH" "$@"
    fi
    
    log_success "Sufficient file system permissions available"
}

# Enhanced startup banner with project information
print_startup_banner() {
    echo "🚀 Starting ${PROJECT_NAME}..."
    echo "📋 Project: ${PROJECT_VERSION} - AI Skills Demonstration"
    echo "🔧 Script: ${SCRIPT_NAME}"
    echo "📅 Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "📍 Location: ai-test/app/"
    echo "----------------------------------------"
}

# Parse command line arguments for GUI selection
parse_arguments() {
    case "${1:-}" in
        --gui=chainlit|chainlit)
            export GUI_MODE="chainlit"
            log_info "GUI mode set to Chainlit via command line"
            ;;
        --gui=gradio|gradio)
            export GUI_MODE="gradio"
            log_info "GUI mode set to Gradio via command line"
            ;;
        --help|-h)
            echo "Usage: $0 [chainlit|gradio|--gui=chainlit|--gui=gradio]"
            echo ""
            echo "Options:"
            echo "  chainlit, --gui=chainlit    Launch with Chainlit GUI"
            echo "  gradio, --gui=gradio        Launch with Gradio GUI"
            echo "  --help, -h                  Show this help message"
            echo ""
            echo "Interactive mode:"
            echo "  $0                          Prompt for GUI selection"
            exit 0
            ;;
        "")
            # No argument provided, will prompt user
            ;;
        *)
            log_warning "Unknown argument: $1. Will prompt for GUI selection."
            ;;
    esac
}

# GUI selection function for interactive mode
select_gui_mode() {
    # Skip if GUI_MODE already set via command line
    if [[ -n "${GUI_MODE:-}" ]]; then
        return 0
    fi
    
    log_info "GUI Selection"
    echo ""
    echo "🎛️  Choose your preferred GUI interface:"
    echo "   1) Chainlit GUI (conversational interface with memory)"
    echo "   2) Gradio GUI (visual interface with memory capabilities)"
    echo ""
    echo -n "Enter your choice (1 or 2) [default: 1]: "
    
    read -r gui_choice
    
    case "${gui_choice}" in
        1|""|chainlit)
            export GUI_MODE="chainlit"
            log_success "Selected: Chainlit GUI"
            ;;
        2|gradio)
            export GUI_MODE="gradio"
            log_success "Selected: Gradio GUI"
            ;;
        *)
            log_warning "Invalid choice. Defaulting to Chainlit GUI"
            export GUI_MODE="chainlit"
            ;;
    esac
    echo ""
}

# Check if we're in the correct directory (ai-test/app/)
check_directory() {
    log_info "Verifying script location..."
    
    # Check if we're in the app directory
    if [[ ! -f "pyproject.toml" ]] || [[ ! -f "ignite.py" ]]; then
        log_error "Script must be run from ai-test/app/ directory"
        echo ""
        echo "📍 Current directory: $(pwd)"
        echo "📋 Expected files: pyproject.toml, ignite.py"
        echo ""
        echo "🔧 To fix this:"
        echo "   cd ai-test/app"
        echo "   bash start.sh"
        exit 1
    fi
    
    log_success "Running from correct directory: ai-test/app/"
}

# Check if virtual environment is activated
check_virtual_environment() {
    log_info "Checking virtual environment..."
    
    # If running as root, check for preserved VIRTUAL_ENV or look for .venv
    if [[ $EUID -eq 0 ]]; then
        if [[ -n "${VIRTUAL_ENV:-}" ]]; then
            log_success "Virtual environment preserved: ${VIRTUAL_ENV}"
            return 0
        elif [[ -d "../.venv" ]]; then
            log_info "Virtual environment detected but not active under sudo"
            log_success "Continuing with root permissions (virtual environment will be managed by Poetry)"
            return 0
        fi
    fi
    
    if [[ -z "${VIRTUAL_ENV:-}" ]]; then
        log_error "Virtual environment not activated"
        echo ""
        echo "🔧 Please activate your virtual environment first:"
        echo "   cd ai-test"
        echo "   source .venv/bin/activate"
        echo "   cd app"
        echo "   bash start.sh"
        exit 1
    fi
    
    log_success "Virtual environment active: ${VIRTUAL_ENV}"
}

# Comprehensive Poetry installation check
check_poetry_installation() {
    log_info "Checking Poetry installation..."
    
    # First try to find poetry in the current PATH
    if command -v poetry &> /dev/null; then
        local poetry_version
        poetry_version=$(poetry --version 2>/dev/null | cut -d' ' -f3 || echo "unknown")
        log_success "Poetry found in PATH (version: ${poetry_version})"
        return 0
    fi
    
    # If not found and we have a virtual environment, try looking there
    if [[ -n "${VIRTUAL_ENV:-}" ]] && [[ -f "${VIRTUAL_ENV}/bin/poetry" ]]; then
        # Add virtual environment to PATH temporarily
        export PATH="${VIRTUAL_ENV}/bin:${PATH}"
        local poetry_version
        poetry_version=$(poetry --version 2>/dev/null | cut -d' ' -f3 || echo "unknown")
        log_success "Poetry found in virtual environment (version: ${poetry_version})"
        return 0
    fi
    
    # Try looking in common virtual environment locations
    local venv_paths=("../.venv/bin/poetry" "./.venv/bin/poetry" "../.venv/bin/poetry")
    for venv_poetry in "${venv_paths[@]}"; do
        if [[ -f "$venv_poetry" ]]; then
            # Add the directory to PATH
            local venv_bin_dir="$(dirname "$venv_poetry")"
            export PATH="${venv_bin_dir}:${PATH}"
            local poetry_version
            poetry_version=$(poetry --version 2>/dev/null | cut -d' ' -f3 || echo "unknown")
            log_success "Poetry found in virtual environment (version: ${poetry_version})"
            return 0
        fi
    done
    
    log_error "Poetry not found. Poetry is required for dependency management."
    echo ""
    echo "📦 Install Poetry using:"
    echo "   pip install poetry"
    echo ""
    echo "After installation, run this script again."
    exit 1
}

# Enhanced Python version validation
validate_python_version() {
    log_info "Validating Python version compatibility..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "python3 command not found. Please install Python 3.9-3.10."
        exit 1
    fi
    
    local python_version
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
    
    echo "🐍 Python version: ${python_version}"
    
    # Validate version range (>=3.9,<3.11)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3,9) and sys.version_info < (3,11) else 1)" 2>/dev/null; then
        log_error "Python version ${python_version} is not compatible."
        echo ""
        echo "📋 Requirements:"
        echo "   - Minimum: Python ${REQUIRED_PYTHON_MIN}"
        echo "   - Maximum: Python ${REQUIRED_PYTHON_MAX} (exclusive)"
        echo "   - Current: Python ${python_version}"
        exit 1
    fi
    
    log_success "Python version ${python_version} is compatible"
}

# Install dependencies following README procedure
install_dependencies() {
    log_info "Installing project dependencies with Poetry..."
    
    if [[ ! -f "pyproject.toml" ]]; then
        log_error "pyproject.toml not found in current directory"
        exit 1
    fi
    
    # Follow README procedure: poetry lock, then poetry install
    log_info "Running poetry lock..."
    if ! poetry lock; then
        log_error "Failed to lock dependencies"
        exit 1
    fi
    
    log_info "Running poetry install..."
    if ! poetry install --no-interaction; then
        log_error "Failed to install dependencies"
        echo ""
        echo "🔧 Troubleshooting steps:"
        echo "   1. Check network connectivity"
        echo "   2. Try: poetry install --verbose"
        exit 1
    fi
    
    log_success "Dependencies installed successfully"
}

# Verify OpenFabric SDK installation
verify_openfabric_installation() {
    log_info "Verifying openfabric-pysdk installation..."
    
    if ! poetry run python3 -c "
import sys
try:
    import openfabric_pysdk
    print('✅ openfabric-pysdk imported successfully')
except ImportError as e:
    print(f'❌ Import failed: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null; then
        log_error "Failed to import openfabric-pysdk"
        echo ""
        echo "🔧 Try: poetry add openfabric-pysdk"
        exit 1
    fi
}

# Verify GUI files exist
verify_gui_files() {
    log_info "Verifying GUI files..."
    
    local gui_files=("chainlit_gui.py" "gradio_gui.py")
    for file in "${gui_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "GUI file not found: $file"
            exit 1
        fi
    done
    
    log_success "All GUI files found"
}

# Check if required ports are available
check_ports() {
    log_info "Checking port availability..."
    
    local ports=(${API_PORT} ${GUI_PORT})
    for port in "${ports[@]}"; do
        if command -v lsof &> /dev/null && lsof -i ":$port" &> /dev/null; then
            log_warning "Port $port is already in use"
            echo "   To free port $port, run: sudo fuser -k ${port}/tcp"
        else
            log_success "Port $port is available"
        fi
    done
}

# Load environment variables if .env exists
load_environment_variables() {
    if [[ -f ".env" ]]; then
        log_info "Loading environment variables from .env file..."
        set -a
        source .env 2>/dev/null || log_warning "Some environment variables failed to load"
        set +a
        log_success "Environment variables loaded"
    else
        log_info "No .env file found (optional)"
    fi
}

# Pre-flight checks
perform_preflight_checks() {
    log_info "Performing pre-flight checks..."
    
    if [[ ! -f "${APPLICATION_ENTRY}" ]]; then
        log_error "Application entry point '${APPLICATION_ENTRY}' not found"
        exit 1
    fi
    
    log_success "Pre-flight checks completed"
}

# Enhanced application startup with GUI selection
start_application() {
    echo ""
    echo "🎯 Starting AI 3D Generator application..."
    echo "🌐 API Server: http://localhost:${API_PORT}"
    echo "📊 Swagger UI: http://localhost:${API_PORT}/swagger-ui/#/"
    
    # Determine GUI mode and set appropriate port/URL
    GUI_MODE=${GUI_MODE:-chainlit}
    
    if [ "$GUI_MODE" = "gradio" ]; then
        echo "🎛️ Gradio GUI: http://localhost:${GUI_PORT}"
    else
        echo "🧠 Chainlit GUI: http://localhost:${GUI_PORT}"
    fi
    
    echo "📁 Output directory: $(pwd)/outputs"
    echo "📝 Log directory: $(pwd)/logs"
    echo "🧠 Memory database: $(pwd)/memory"
    echo ""
    echo "💡 Usage tips:"
    echo "   - Monitor application logs in the logs/ directory"
    echo "   - Generated files will be saved in outputs/ directory"
    echo "   - Memory system stores all generations for recall"
    if [ "$GUI_MODE" = "gradio" ]; then
        echo "   - Using Gradio visual interface with memory capabilities"
    else
        echo "   - Using Chainlit conversational interface with memory"
    fi
    echo ""
    echo "Press Ctrl+C to stop the application"
    echo "========================================"
    echo ""
    
    # Set up signal handling for graceful shutdown
    trap 'echo ""; log_info "Shutting down all processes..."; kill $API_PID $GUI_PID 2>/dev/null; exit 0' INT TERM
    
    # Start the API server (ignite.py)
    log_info "Starting core engine (ignite.py) on port ${API_PORT}..."
    poetry run python3 "${APPLICATION_ENTRY}" &
    API_PID=$!
    
    # Give API server time to start
    sleep 15
    
    # Start the selected GUI
    if [ "$GUI_MODE" = "gradio" ]; then
        log_info "Launching Gradio GUI on port ${GUI_PORT}..."
        poetry run python3 gradio_gui.py &
        GUI_PID=$!
    else
        log_info "Launching Chainlit GUI on port ${GUI_PORT}..."
        poetry run chainlit run chainlit_gui.py -w --port ${GUI_PORT} &
        GUI_PID=$!
    fi
    
    # Wait for both processes
    wait $API_PID $GUI_PID
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
    
    # Parse command line arguments first
    parse_arguments "$@"
    
    # Execute startup sequence
    print_startup_banner
    check_and_elevate_permissions
    check_directory
    check_virtual_environment
    check_poetry_installation
    validate_python_version
    install_dependencies
    verify_openfabric_installation
    verify_gui_files
    load_environment_variables
    check_ports
    perform_preflight_checks
    select_gui_mode
    start_application
}

# Execute main function with all arguments
main "$@"