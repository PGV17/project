#!/usr/bin/env python3
"""
Quick deployment helper for Financial Analysis App
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_git_status():
    """Check if we're in a git repository"""
    return os.path.exists('.git')

def deploy_streamlit_cloud():
    """Guide for Streamlit Cloud deployment"""
    print("\n🌟 Streamlit Cloud Deployment Guide")
    print("=" * 50)
    
    if not check_git_status():
        print("📝 Initializing Git repository...")
        run_command("git init", "Git initialization")
        run_command("git add .", "Adding files to git")
        run_command('git commit -m "Initial commit: Financial analysis app"', "Initial commit")
    
    print("\n📋 Next steps:")
    print("1. Create a GitHub repository named 'financial-analysis-app'")
    print("2. Run: git remote add origin https://github.com/yourusername/financial-analysis-app.git")
    print("3. Run: git push -u origin main")
    print("4. Go to https://share.streamlit.io")
    print("5. Connect your GitHub account")
    print("6. Select your repository and deploy!")
    print("\n🎉 Your app will be live at: https://yourusername-financial-analysis-app-app-xxxxx.streamlit.app")

def deploy_docker():
    """Deploy using Docker"""
    print("\n🐳 Docker Deployment")
    print("=" * 30)
    
    if run_command("docker --version", "Checking Docker installation"):
        print("🔨 Building Docker image...")
        if run_command("docker build -t financial-analysis-app .", "Building Docker image"):
            print("🚀 Starting container...")
            run_command("docker run -d -p 8501:8501 --name financial-app financial-analysis-app", "Starting container")
            print("\n✅ App is running at: http://localhost:8501")
            print("📋 To stop: docker stop financial-app")
            print("📋 To remove: docker rm financial-app")
    else:
        print("❌ Docker not found. Please install Docker first.")

def deploy_local():
    """Run locally with optimal settings"""
    print("\n💻 Local Development Server")
    print("=" * 35)
    
    print("🔄 Starting Streamlit with production settings...")
    command = "streamlit run app.py --server.headless true --browser.gatherUsageStats false"
    subprocess.run(command, shell=True)

def main():
    print("🚀 Financial Analysis App - Deployment Helper")
    print("=" * 55)
    print("\nChoose your deployment option:")
    print("1. 🌟 Streamlit Cloud (FREE - Recommended)")
    print("2. 🐳 Docker (Local container)")
    print("3. 💻 Local development server")
    print("4. 📖 View deployment guide")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            deploy_streamlit_cloud()
            break
        elif choice == "2":
            deploy_docker()
            break
        elif choice == "3":
            deploy_local()
            break
        elif choice == "4":
            print("\n📖 Opening deployment guide...")
            print("See DEPLOYMENT.md for detailed instructions")
            break
        else:
            print("❌ Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
