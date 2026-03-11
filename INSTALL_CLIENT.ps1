# ==============================================================================
# Twinkle Client Installation Script for Windows
#
# This script sets up a Python environment for using Twinkle client with Tinker.
# It will:
#   1. Check if conda is installed; if not, download and install Miniconda
#   2. Create a new conda environment with Python 3.11
#   3. Install twinkle-kit with tinker dependencies
#
# Usage (run in PowerShell):
#   .\INSTALL_CLIENT.ps1 [ENV_NAME]
#
# Arguments:
#   ENV_NAME  - Name of the conda environment (default: twinkle-client)
#
# After installation, activate the environment with:
#   conda activate twinkle-client
# ==============================================================================

param(
    [string]$EnvName = "twinkle-client"
)

$ErrorActionPreference = "Stop"
$PythonVersion = "3.11"
$MinicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Twinkle Client Installation (Windows)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Environment name: $EnvName"
Write-Host "Python version: $PythonVersion"
Write-Host ""

# ==============================================================================
# Step 1: Check and install Conda
# ==============================================================================

function Test-CondaInstalled {
    try {
        $condaVersion = conda --version 2>$null
        if ($condaVersion) {
            Write-Host "[OK] Conda is already installed: $condaVersion" -ForegroundColor Green
            return $true
        }
    } catch {}

    # Check common installation paths
    $condaPaths = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe",
        "C:\ProgramData\Anaconda3\Scripts\conda.exe"
    )

    foreach ($path in $condaPaths) {
        if (Test-Path $path) {
            $condaDir = Split-Path (Split-Path $path)
            Write-Host "[!] Found conda at: $condaDir" -ForegroundColor Yellow
            Write-Host "    Adding to PATH for this session..."
            $env:PATH = "$condaDir\Scripts;$condaDir;$env:PATH"
            return $true
        }
    }

    Write-Host "[!] Conda not found" -ForegroundColor Yellow
    return $false
}

function Install-Miniconda {
    Write-Host ""
    Write-Host "Installing Miniconda..." -ForegroundColor Cyan

    $installerPath = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"
    $installDir = "$env:USERPROFILE\miniconda3"

    Write-Host "Downloading Miniconda from: $MinicondaUrl"

    # Download installer
    try {
        Invoke-WebRequest -Uri $MinicondaUrl -OutFile $installerPath -UseBasicParsing
    } catch {
        Write-Host "[ERROR] Failed to download Miniconda: $_" -ForegroundColor Red
        exit 1
    }

    Write-Host "Installing Miniconda to: $installDir"
    Write-Host "This may take a few minutes..."

    # Run installer silently
    Start-Process -FilePath $installerPath -ArgumentList @(
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=1",
        "/S",
        "/D=$installDir"
    ) -Wait -NoNewWindow

    # Add to PATH for current session
    $env:PATH = "$installDir\Scripts;$installDir;$env:PATH"

    # Clean up
    Remove-Item $installerPath -Force -ErrorAction SilentlyContinue

    Write-Host "[OK] Miniconda installed successfully" -ForegroundColor Green
    Write-Host ""
    Write-Host "[!] IMPORTANT: Restart PowerShell after installation to use conda globally" -ForegroundColor Yellow
}

if (-not (Test-CondaInstalled)) {
    $response = Read-Host "Do you want to install Miniconda? [Y/n]"
    if ($response -match "^[Nn]") {
        Write-Host "Installation cancelled. Please install conda manually."
        exit 1
    }
    Install-Miniconda
}

# Initialize conda for PowerShell
try {
    $condaHook = conda shell.powershell hook 2>$null | Out-String
    if ($condaHook) {
        Invoke-Expression $condaHook
    }
} catch {}

# ==============================================================================
# Step 2: Create conda environment
# ==============================================================================

Write-Host ""
Write-Host "Creating conda environment: $EnvName (Python $PythonVersion)..." -ForegroundColor Cyan

# Accept Conda ToS for default channels (required for conda >= 26.x)
Write-Host "Accepting Conda Terms of Service for default channels..."
try {
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>$null
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>$null
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>$null
} catch {
    # Older conda versions don't have tos command, ignore
}

# Check if environment already exists
$envList = conda env list 2>$null
if ($envList -match "^$EnvName\s") {
    Write-Host "[!] Environment '$EnvName' already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to remove and recreate it? [y/N]"
    if ($response -match "^[Yy]") {
        Write-Host "Removing existing environment..."
        conda env remove -n $EnvName -y
    } else {
        Write-Host "Using existing environment..."
    }
}

# Create environment if it doesn't exist
$envList = conda env list 2>$null
if (-not ($envList -match "^$EnvName\s")) {
    Write-Host "Running: conda create -n $EnvName python=$PythonVersion -y"
    conda create -n $EnvName python=$PythonVersion -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create conda environment. Exit code: $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

# Verify environment was created
$envList = conda env list 2>$null
if (-not ($envList -match "^$EnvName\s")) {
    Write-Host "[ERROR] Environment '$EnvName' was not created successfully." -ForegroundColor Red
    Write-Host "Please run manually:" -ForegroundColor Yellow
    Write-Host "  conda create -n $EnvName python=$PythonVersion" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Environment '$EnvName' is ready" -ForegroundColor Green

# ==============================================================================
# Step 3: Install dependencies
# ==============================================================================

Write-Host ""
Write-Host "Activating environment and installing dependencies..." -ForegroundColor Cyan

# Activate environment
Write-Host "Activating environment '$EnvName'..."
try {
    conda activate $EnvName
    if ($LASTEXITCODE -ne 0) {
        throw "conda activate failed"
    }
} catch {
    Write-Host "[!] Standard activation failed, trying alternative method..." -ForegroundColor Yellow
    # Alternative: run commands in the conda environment directly
    $condaBase = (conda info --base 2>$null).Trim()
    $activateScript = Join-Path $condaBase "Scripts\activate.bat"
    if (Test-Path $activateScript) {
        cmd /c "call `"$activateScript`" $EnvName && pip --version" 2>$null
    }
}

# Verify we're in the correct environment
$currentPython = python --version 2>&1
Write-Host "Current Python: $currentPython"
if ($currentPython -notmatch "3\.11") {
    Write-Host "[!] Warning: Python version mismatch. Expected 3.11, got: $currentPython" -ForegroundColor Yellow
    Write-Host "Attempting to use conda run instead..." -ForegroundColor Yellow
    $UseCondaRun = $true
} else {
    $UseCondaRun = $false
}

# Upgrade pip
if ($UseCondaRun) {
    conda run -n $EnvName pip install --upgrade pip
} else {
    pip install --upgrade pip
}

# Check if we're in the twinkle source directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if (Test-Path "$ScriptDir\pyproject.toml") {
    Write-Host ""
    Write-Host "Installing twinkle from source (with tinker support)..." -ForegroundColor Cyan
    if ($UseCondaRun) {
        conda run -n $EnvName pip install -e "$ScriptDir[tinker]"
    } else {
        pip install -e "$ScriptDir[tinker]"
    }
} else {
    Write-Host ""
    Write-Host "Installing twinkle-kit from PyPI (with tinker support)..." -ForegroundColor Cyan
    if ($UseCondaRun) {
        conda run -n $EnvName pip install "twinkle-kit[tinker]"
    } else {
        pip install "twinkle-kit[tinker]"
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install twinkle. Exit code: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# ==============================================================================
# Step 4: Verify installation
# ==============================================================================

Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Cyan
Write-Host ""

$verifyScript = @"
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
    print('  [OK] init_tinker_client available')
except ImportError as e:
    print(f'  [FAIL] init_tinker_client: {e}')

try:
    from twinkle.dataloader import DataLoader
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import SelfCognitionProcessor
    print('  [OK] twinkle core components available')
except ImportError as e:
    print(f'  [FAIL] twinkle core: {e}')
"@

if ($UseCondaRun) {
    conda run -n $EnvName python -c $verifyScript
} else {
    python -c $verifyScript
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To activate the environment, run:"
Write-Host "  conda activate $EnvName" -ForegroundColor Yellow
Write-Host ""
Write-Host "Example usage (see cookbook/client/tinker/):"
Write-Host '  $env:MODELSCOPE_TOKEN = "your-token"' -ForegroundColor Yellow
Write-Host "  python cookbook/client/tinker/modelscope_service/self_cognition.py" -ForegroundColor Yellow
Write-Host ""
