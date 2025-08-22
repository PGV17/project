#!/usr/bin/env python3
"""
Setup verification script for Financial Analysis Application
Run this after installation to verify everything is working correctly.
"""

import sys
import importlib
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Compatible)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Requires Python 3.8+)")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name} {version}")
        return True
    except ImportError:
        print(f"❌ {package_name} (Not installed)")
        return False

def check_application_modules():
    """Check if application modules can be imported"""
    modules = ['app', 'ml_models', 'nlp_engine']
    success = True
    
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✅ {module}.py")
        except ImportError as e:
            print(f"❌ {module}.py ({str(e)})")
            success = False
    
    return success

def main():
    print("🔍 Financial Analysis Application - Setup Verification")
    print("=" * 60)
    
    # Check Python version
    print("\n📋 Python Version Check:")
    python_ok = check_python_version()
    
    # Check required packages
    print("\n📦 Package Dependencies:")
    packages = [
        ('streamlit', 'streamlit'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('yfinance', 'yfinance'),
        ('scikit-learn', 'sklearn'),
        ('requests', 'requests'),
        ('nltk', 'nltk'),
        ('textblob', 'textblob')
    ]
    
    packages_ok = all(check_package(name, import_name) for name, import_name in packages)
    
    # Check application modules
    print("\n🐍 Application Modules:")
    modules_ok = check_application_modules()
    
    # Download NLTK data if needed
    print("\n📚 NLTK Data:")
    try:
        import nltk
        required_data = ['vader_lexicon', 'punkt', 'stopwords', 'wordnet']
        for data in required_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
                print(f"✅ {data}")
            except LookupError:
                print(f"📥 Downloading {data}...")
                nltk.download(data, quiet=True)
                print(f"✅ {data} (Downloaded)")
    except ImportError:
        print("❌ NLTK not available")
    
    # Final status
    print("\n" + "=" * 60)
    if python_ok and packages_ok and modules_ok:
        print("🎉 Setup verification completed successfully!")
        print("\n🚀 You can now run the application with:")
        print("   streamlit run app.py")
    else:
        print("❌ Setup verification failed. Please check the issues above.")
        print("\n💡 Try installing missing packages with:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
