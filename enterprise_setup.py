"""
Enterprise Setup Script for Dinesh Trading Dashboard.
Installs all dependencies and configures the advanced AI trading system.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path
import shutil


class EnterpriseSetup:
    """Enterprise setup and configuration manager."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.system_info = {
            'platform': platform.system(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        }
        
        # Required packages for enterprise features
        self.required_packages = [
            # Core dependencies
            'dash>=2.14.0',
            'plotly>=5.15.0',
            'pandas>=2.0.0',
            'numpy>=1.24.0',
            'scipy>=1.10.0',
            'scikit-learn>=1.3.0',
            
            # Advanced NLP and ML
            'torch>=2.0.0',
            'transformers>=4.30.0',
            'xgboost>=1.7.0',
            'lightgbm>=4.0.0',
            
            # Performance optimization
            'psutil>=5.9.0',
            'numba>=0.57.0',
            
            # Network analysis
            'networkx>=3.1.0',
            
            # Data processing
            'aiohttp>=3.8.0',
            'python-dotenv>=1.0.0',
            'joblib>=1.3.0',
            
            # Financial data
            'yfinance>=0.2.0',
            'alpha-vantage>=2.3.0',
            
            # Social media APIs
            'tweepy>=4.14.0',
            'praw>=7.7.0',
            'newsapi-python>=0.2.6',
            
            # Optional language detection
            'langdetect>=1.0.9',
            
            # PDF generation
            'reportlab>=4.0.0'
        ]
        
        # GPU-specific packages
        self.gpu_packages = [
            'torch-audio',
            'torch-vision'
        ]
        
        # Development packages
        self.dev_packages = [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0'
        ]
    
    def check_system_requirements(self):
        """Check system requirements."""
        print("🔍 Checking System Requirements...")
        print("=" * 50)
        
        # Python version check
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python 3.8+ required. Current version:", sys.version)
            return False
        else:
            print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Platform info
        print(f"✅ Platform: {self.system_info['platform']} ({self.system_info['architecture']})")
        
        # Memory check
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 4:
                print(f"⚠️  Low memory: {memory_gb:.1f}GB (8GB+ recommended)")
            else:
                print(f"✅ Memory: {memory_gb:.1f}GB")
        except ImportError:
            print("⚠️  Cannot check memory (psutil not installed)")
        
        # GPU check
        gpu_available = self.check_gpu_availability()
        if gpu_available:
            print("✅ GPU acceleration available")
        else:
            print("⚠️  No GPU detected (CPU-only mode)")
        
        return True
    
    def check_gpu_availability(self):
        """Check GPU availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def install_packages(self, include_gpu=False, include_dev=False):
        """Install required packages."""
        print("\n📦 Installing Required Packages...")
        print("=" * 50)
        
        packages_to_install = self.required_packages.copy()
        
        if include_gpu and self.check_gpu_availability():
            packages_to_install.extend(self.gpu_packages)
            print("🎮 Including GPU packages...")
        
        if include_dev:
            packages_to_install.extend(self.dev_packages)
            print("🛠️  Including development packages...")
        
        # Install packages
        for package in packages_to_install:
            try:
                print(f"Installing {package}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package, '--upgrade'
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✅ {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
        
        print("\n✅ All packages installed successfully!")
        return True
    
    def setup_environment(self):
        """Setup environment configuration."""
        print("\n🔧 Setting Up Environment...")
        print("=" * 50)
        
        # Create .env file if it doesn't exist
        env_file = Path('.env')
        env_example = Path('.env.example')
        
        if not env_file.exists() and env_example.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your API keys")
        elif env_file.exists():
            print("✅ .env file already exists")
        else:
            self.create_default_env_file()
        
        # Create logs directory
        logs_dir = Path('logs')
        if not logs_dir.exists():
            logs_dir.mkdir()
            print("✅ Created logs directory")
        
        # Create models directory for saved ML models
        models_dir = Path('models')
        if not models_dir.exists():
            models_dir.mkdir()
            print("✅ Created models directory")
        
        return True
    
    def create_default_env_file(self):
        """Create default .env file."""
        env_content = """# Dinesh Enterprise Trading Dashboard Configuration

# Twitter API Configuration
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# Reddit API Configuration
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=Dinesh Enterprise Trading Bot v2.0

# News API Configuration
NEWS_API_KEY=your_news_api_key_here

# Financial Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
FINNHUB_API_KEY=your_finnhub_key_here

# System Configuration
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
LOG_FILE=logs/enterprise_dinesh.log

# Database Configuration
DATABASE_URL=sqlite:///enterprise_goquant.db

# AI Configuration
USE_GPU=auto
NLP_MODEL=ProsusAI/finbert
ENABLE_SARCASM_DETECTION=true
ENABLE_MULTILANG=true

# Performance Configuration
BATCH_SIZE=200
BUFFER_SIZE=2000
CACHE_SIZE=5000
SIMD_ENABLED=true

# Asset Configuration
DEFAULT_ASSETS=BTC,ETH,SPY,AAPL,TSLA,NVDA,MSFT,GOOGL
CRYPTO_ASSETS=BTC,ETH,ADA,SOL,MATIC,DOT,AVAX,LINK
STOCK_ASSETS=SPY,QQQ,AAPL,MSFT,GOOGL,TSLA,NVDA,AMZN,META,NFLX

# Dashboard Configuration
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8050
DASHBOARD_DEBUG=false
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
        
        print("✅ Created default .env file")
    
    def run_tests(self):
        """Run comprehensive test suite."""
        print("\n🧪 Running Comprehensive Tests...")
        print("=" * 50)
        
        try:
            # Run the test suite
            result = subprocess.run([
                sys.executable, 'comprehensive_test_suite.py'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed!")
                print(result.stdout)
                return True
            else:
                print("❌ Some tests failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def create_startup_script(self):
        """Create startup script for easy launching."""
        print("\n📝 Creating Startup Scripts...")
        print("=" * 50)
        
        # Windows batch file
        if self.system_info['platform'] == 'Windows':
            batch_content = """@echo off
echo Starting Dinesh Enterprise Trading Dashboard...
echo.
python enterprise_dashboard.py
pause
"""
            with open('start_enterprise.bat', 'w') as f:
                f.write(batch_content)
            print("✅ Created start_enterprise.bat")
        
        # Unix shell script
        else:
            shell_content = """#!/bin/bash
echo "Starting Dinesh Enterprise Trading Dashboard..."
echo
python enterprise_dashboard.py
"""
            with open('start_enterprise.sh', 'w') as f:
                f.write(shell_content)
            
            # Make executable
            os.chmod('start_enterprise.sh', 0o755)
            print("✅ Created start_enterprise.sh")
        
        # Python launcher script
        launcher_content = """#!/usr/bin/env python3
\"\"\"
Enterprise Dashboard Launcher
\"\"\"

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import and run dashboard
from enterprise_dashboard import EnterpriseDashboard

if __name__ == "__main__":
    print("🚀 Launching Dinesh Enterprise Trading Dashboard...")
    dashboard = EnterpriseDashboard()
    dashboard.run()
"""
        
        with open('launch_enterprise.py', 'w') as f:
            f.write(launcher_content)
        
        print("✅ Created launch_enterprise.py")
    
    def display_completion_message(self):
        """Display setup completion message."""
        print("\n" + "=" * 80)
        print("🎉 DINESH ENTERPRISE TRADING DASHBOARD SETUP COMPLETE!")
        print("=" * 80)
        
        print("\n🚀 ENTERPRISE FEATURES INSTALLED:")
        print("   ✅ Advanced NLP with FinBERT and Transformer models")
        print("   ✅ Machine Learning ensemble predictions (XGBoost, LightGBM, Neural Networks)")
        print("   ✅ Performance optimization with SIMD and memory pools")
        print("   ✅ Market psychology and behavioral bias detection")
        print("   ✅ Cross-market correlation and sentiment contagion analysis")
        print("   ✅ Alternative data integration framework")
        print("   ✅ Real-time streaming processing pipeline")
        print("   ✅ Comprehensive monitoring and analytics")
        
        print("\n📋 NEXT STEPS:")
        print("   1. Edit .env file with your API keys")
        print("   2. Run: python comprehensive_test_suite.py")
        print("   3. Launch: python enterprise_dashboard.py")
        print("   4. Open: http://localhost:8050")
        
        print("\n🎯 QUICK START:")
        if self.system_info['platform'] == 'Windows':
            print("   • Double-click: start_enterprise.bat")
        else:
            print("   • Run: ./start_enterprise.sh")
        print("   • Or run: python launch_enterprise.py")
        
        print("\n📚 DOCUMENTATION:")
        print("   • Technical docs: All modules have comprehensive docstrings")
        print("   • API reference: Check individual module files")
        print("   • Performance guide: See performance_optimizer.py")
        print("   • Analytics guide: See advanced_analytics.py")
        
        print("\n⚠️  IMPORTANT:")
        print("   • Configure API keys in .env file before first run")
        print("   • Ensure 8GB+ RAM for optimal performance")
        print("   • GPU recommended for large-scale NLP processing")
        print("   • Monitor system resources during operation")
        
        print("\n🆘 SUPPORT:")
        print("   • Check logs/ directory for error messages")
        print("   • Run tests to verify installation")
        print("   • Review module documentation for troubleshooting")


def main():
    """Main setup function."""
    print("🚀 DINESH ENTERPRISE TRADING DASHBOARD SETUP")
    print("=" * 80)
    print("Advanced AI • Deep Learning • Market Psychology • Performance Optimization")
    print()
    
    setup = EnterpriseSetup()
    
    # Check system requirements
    if not setup.check_system_requirements():
        print("\n❌ System requirements not met. Please upgrade your system.")
        return False
    
    # Ask user for installation options
    print("\n📋 Installation Options:")
    include_gpu = input("Install GPU packages? (y/N): ").lower().startswith('y')
    include_dev = input("Install development packages? (y/N): ").lower().startswith('y')
    run_tests = input("Run tests after installation? (Y/n): ").lower() != 'n'
    
    # Install packages
    if not setup.install_packages(include_gpu=include_gpu, include_dev=include_dev):
        print("\n❌ Package installation failed. Please check your internet connection.")
        return False
    
    # Setup environment
    if not setup.setup_environment():
        print("\n❌ Environment setup failed.")
        return False
    
    # Create startup scripts
    setup.create_startup_script()
    
    # Run tests if requested
    if run_tests:
        test_success = setup.run_tests()
        if not test_success:
            print("\n⚠️  Some tests failed, but installation is complete.")
    
    # Display completion message
    setup.display_completion_message()
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
    
    input("\nPress Enter to exit...")
