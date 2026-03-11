#!/bin/bash
# ==============================================================================
# Twinkle Client Installation Script
#
# This script sets up a Python environment for using Twinkle client with Tinker.
# It will:
#   1. Check if conda is installed; if not, download and install Miniconda
#   2. Create a new conda environment with Python 3.11
#   3. Install twinkle-kit with tinker dependencies
#
# Usage:
#   chmod +x INSTALL_CLIENT.sh
#   ./INSTALL_CLIENT.sh [ENV_NAME]
#
# Arguments:
#   ENV_NAME  - Name of the conda environment (default: twinkle-client)
#
# After installation, activate the environment with:
#   conda activate twinkle-client
# ==============================================================================

set -e  # Exit immediately on error

# Configuration
ENV_NAME="${1:-twinkle-client}"
PYTHON_VERSION="3.11"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest"

echo "=========================================="
echo "Twinkle Client Installation"
echo "=========================================="
echo "Environment name: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""

# ==============================================================================
# Step 1: Check and install Conda
# ==============================================================================

check_conda() {
    if command -v conda &> /dev/null; then
        echo "[✓] Conda is already installed: $(conda --version)"
        return 0
    else
        echo "[!] Conda not found"
        return 1
    fi
}

install_miniconda() {
    echo ""
    echo "Installing Miniconda..."

    # Detect OS and architecture
    OS_TYPE=$(uname -s)
    ARCH=$(uname -m)

    case "$OS_TYPE" in
        Linux)
            INSTALLER_URL="${MINICONDA_URL}-Linux-${ARCH}.sh"
            ;;
        Darwin)
            if [ "$ARCH" = "arm64" ]; then
                INSTALLER_URL="${MINICONDA_URL}-MacOSX-arm64.sh"
            else
                INSTALLER_URL="${MINICONDA_URL}-MacOSX-x86_64.sh"
            fi
            ;;
        *)
            echo "[ERROR] Unsupported OS: $OS_TYPE"
            echo "Please install Miniconda manually from: https://docs.conda.io/en/latest/miniconda.html"
            exit 1
            ;;
    esac

    echo "Downloading Miniconda from: $INSTALLER_URL"

    # Download installer
    INSTALLER_PATH="/tmp/miniconda_installer.sh"
    if command -v curl &> /dev/null; then
        curl -fsSL "$INSTALLER_URL" -o "$INSTALLER_PATH"
    elif command -v wget &> /dev/null; then
        wget -q "$INSTALLER_URL" -O "$INSTALLER_PATH"
    else
        echo "[ERROR] Neither curl nor wget found. Please install one of them."
        exit 1
    fi

    # Run installer
    CONDA_INSTALL_DIR="$HOME/miniconda3"
    echo "Installing Miniconda to: $CONDA_INSTALL_DIR"
    bash "$INSTALLER_PATH" -b -p "$CONDA_INSTALL_DIR"

    # Initialize conda
    "$CONDA_INSTALL_DIR/bin/conda" init bash zsh 2>/dev/null || true

    # Add to current session
    export PATH="$CONDA_INSTALL_DIR/bin:$PATH"

    # Clean up
    rm -f "$INSTALLER_PATH"

    echo "[✓] Miniconda installed successfully"
    echo ""
    echo "[!] IMPORTANT: Restart your shell or run:"
    echo "    source ~/.bashrc  # or ~/.zshrc"
}

if ! check_conda; then
    read -p "Do you want to install Miniconda? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        echo "Installation cancelled. Please install conda manually."
        exit 1
    fi
    install_miniconda
fi

# Ensure conda command is available
eval "$(conda shell.bash hook 2>/dev/null)" || true

# ==============================================================================
# Step 2: Create conda environment
# ==============================================================================

echo ""
echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)..."

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[!] Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "Using existing environment..."
    fi
fi

# Create environment if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

echo "[✓] Environment '$ENV_NAME' is ready"

# ==============================================================================
# Step 3: Install dependencies
# ==============================================================================

echo ""
echo "Activating environment and installing dependencies..."

# Activate environment
conda activate "$ENV_NAME"

# Upgrade pip
pip install --upgrade pip

# Check if we're in the twinkle source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    echo ""
    echo "Installing twinkle from source (with tinker support)..."
    pip install -e "$SCRIPT_DIR[tinker]"
else
    echo ""
    echo "Installing twinkle-kit from PyPI (with tinker support)..."
    pip install 'twinkle-kit[tinker]'
fi

# ==============================================================================
# Step 4: Verify installation
# ==============================================================================

echo ""
echo "Verifying installation..."
echo ""

python -c "
import sys
print(f'Python: {sys.version}')
print()

packages = [
    'twinkle',
    'twinkle_client',
    'tinker',
    'transformers',
    'peft',
    'modelscope',
    'datasets',
]

print('Installed packages:')
print('-' * 40)

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg}: {version}')
    except ImportError as e:
        print(f'  {pkg}: NOT INSTALLED ({e})')

print()
print('Testing twinkle client imports...')
try:
    from twinkle_client import init_tinker_client
    print('  [✓] init_tinker_client available')
except ImportError as e:
    print(f'  [✗] init_tinker_client: {e}')

try:
    from twinkle.dataloader import DataLoader
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import SelfCognitionProcessor
    print('  [✓] twinkle core components available')
except ImportError as e:
    print(f'  [✗] twinkle core: {e}')
"

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Example usage (see cookbook/client/tinker/):"
echo "  export MODELSCOPE_TOKEN='your-token'"
echo "  python cookbook/client/tinker/modelscope_service/self_cognition.py"
echo ""
