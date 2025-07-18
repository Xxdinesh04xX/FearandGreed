"""
Installation script for GoQuant Sentiment Trader.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {command}")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main installation process."""
    print("GoQuant Sentiment Trader Installation")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("✗ Python 3.9 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create virtual environment
    if not run_command("python -m venv venv", "Creating virtual environment"):
        sys.exit(1)
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Upgrade pip
    if not run_command(f"{pip_command} install --upgrade pip", "Upgrading pip"):
        sys.exit(1)
    
    # Install dependencies
    if not run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Install package in development mode
    if not run_command(f"{pip_command} install -e .", "Installing GoQuant package"):
        sys.exit(1)
    
    # Create necessary directories
    directories = ["logs", "data", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Copy environment file
    if not Path(".env").exists():
        if Path(".env.example").exists():
            import shutil
            shutil.copy(".env.example", ".env")
            print("✓ Created .env file from .env.example")
            print("  Please edit .env file with your API keys")
        else:
            print("⚠ .env.example not found, please create .env file manually")
    
    print("\n" + "=" * 40)
    print("Installation completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Run the application:")
    print("   python -m goquant.main")
    print("   or")
    print("   goquant-sentiment run")
    print("\nFor help:")
    print("   goquant-sentiment --help")


if __name__ == "__main__":
    main()
