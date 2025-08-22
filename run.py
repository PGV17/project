#!/usr/bin/env python3
"""
CredTech Intelligence Platform Runner
Simple script to start the application with proper setup
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        sys.exit(1)

def download_nltk_data():
    """Download required NLTK data"""
    print("🔤 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ NLTK data download failed: {e}")

def run_application():
    """Start the Streamlit application"""
    print("🚀 Starting CredTech Intelligence Platform...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")

def main():
    """Main runner function"""
    print("🏦 CredTech Intelligence Platform Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Check if requirements need to be installed
    try:
        import streamlit
        import pandas
        import plotly
        print("✅ Dependencies already installed")
    except ImportError:
        install_requirements()
    
    # Download NLTK data
    download_nltk_data()
    
    # Run the application
    run_application()

if __name__ == "__main__":
    main()
