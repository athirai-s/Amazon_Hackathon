#!/usr/bin/env python3
"""
Setup script for EcoAesthetics Python backend
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    if not run_command("python -m venv venv", "Creating virtual environment"):
        return False
    
    return True

def install_dependencies():
    """Install Python dependencies"""
    # Determine the correct pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        pip_path = "venv/bin/pip"
    
    commands = [
        f"{pip_path} install --upgrade pip",
        f"{pip_path} install -r requirements.txt"
    ]
    
    for command in commands:
        if not run_command(command, f"Running: {command}"):
            return False
    
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("🔄 Testing package imports...")
    
    # Determine the correct python path based on OS
    if os.name == 'nt':  # Windows
        python_path = "venv\\Scripts\\python"
    else:  # Unix/Linux/macOS
        python_path = "venv/bin/python"
    
    test_script = """
import torch
import torchvision
import transformers
import fastapi
import uvicorn
import PIL
import numpy
import cv2
print("✅ All packages imported successfully!")
"""
    
    try:
        result = subprocess.run(
            [python_path, "-c", test_script], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Import test failed: {e.stderr}")
        return False

def create_startup_scripts():
    """Create convenient startup scripts"""
    
    # Windows batch script
    windows_script = """@echo off
echo Starting EcoAesthetics Backend Server...
cd /d "%~dp0"
call venv\\Scripts\\activate
python main.py
pause
"""
    
    with open("start_server.bat", "w") as f:
        f.write(windows_script)
    
    # Unix shell script
    unix_script = """#!/bin/bash
echo "Starting EcoAesthetics Backend Server..."
cd "$(dirname "$0")"
source venv/bin/activate
python main.py
"""
    
    with open("start_server.sh", "w") as f:
        f.write(unix_script)
    
    # Make shell script executable
    if os.name != 'nt':
        os.chmod("start_server.sh", 0o755)
    
    print("✅ Startup scripts created (start_server.bat / start_server.sh)")

def main():
    """Main setup function"""
    print("🚀 Setting up EcoAesthetics Python Backend")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup virtual environment
    if not setup_virtual_environment():
        print("❌ Failed to set up virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("❌ Package import test failed")
        sys.exit(1)
    
    # Create startup scripts
    create_startup_scripts()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the backend server:")
    if os.name == 'nt':
        print("   Windows: double-click start_server.bat")
        print("   Or run: venv\\Scripts\\activate && python main.py")
    else:
        print("   Unix/Linux/macOS: ./start_server.sh")
        print("   Or run: source venv/bin/activate && python main.py")
    
    print("\n2. Start the React frontend (in another terminal):")
    print("   cd ../ecoaesthetics")
    print("   npm run dev")
    
    print("\n3. Open http://localhost:3000 in your browser")
    print("\nThe backend will be available at http://localhost:8000")

if __name__ == "__main__":
    main()
