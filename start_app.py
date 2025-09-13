#!/usr/bin/env python3
"""
Startup script for Plant Disease Detection App
This script starts both the FastAPI backend and React frontend
"""

import subprocess
import sys
import os
import time
import threading
import webbrowser
from pathlib import Path

def run_command(command, cwd=None, shell=True):
    """Run a command in a subprocess"""
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            shell=shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"Error running command '{command}': {e}")
        return None

def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting FastAPI Backend...")
    backend_dir = Path("backend/api")
    
    if not backend_dir.exists():
        print(f"❌ Backend directory not found: {backend_dir}")
        return None
    
    # Check if requirements are installed
    try:
        import fastapi
        import uvicorn
        import tensorflow
    except ImportError:
        print("📦 Installing backend dependencies...")
        install_process = run_command("pip install -r requirements.txt", cwd=backend_dir)
        if install_process:
            install_process.wait()
    
    # Start the backend
    backend_process = run_command("python app.py", cwd=backend_dir)
    
    if backend_process:
        print("✅ Backend started successfully!")
        print("   API available at: http://localhost:8000")
        return backend_process
    else:
        print("❌ Failed to start backend")
        return None

def start_frontend():
    """Start the React frontend"""
    print("🎨 Starting React Frontend...")
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return None
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        install_process = run_command("npm install", cwd=frontend_dir)
        if install_process:
            install_process.wait()
            print("✅ Frontend dependencies installed!")
    
    # Start the frontend
    frontend_process = run_command("npm start", cwd=frontend_dir)
    
    if frontend_process:
        print("✅ Frontend started successfully!")
        print("   App available at: http://localhost:3000")
        return frontend_process
    else:
        print("❌ Failed to start frontend")
        return None

def wait_for_backend():
    """Wait for backend to be ready"""
    import requests
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ Backend is ready!")
                return True
        except:
            pass
        time.sleep(2)
        print(f"⏳ Waiting for backend... ({attempt + 1}/{max_attempts})")
    
    print("❌ Backend failed to start within timeout")
    return False

def main():
    """Main function to start the application"""
    print("🌱 Plant Disease Detection App")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("backend").exists() or not Path("frontend").exists():
        print("❌ Please run this script from the project root directory")
        print("   Make sure both 'backend' and 'frontend' folders exist")
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Cannot start app without backend")
        sys.exit(1)
    
    # Wait for backend to be ready
    if not wait_for_backend():
        print("❌ Backend is not responding")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Cannot start app without frontend")
        backend_process.terminate()
        sys.exit(1)
    
    # Wait a bit for frontend to start
    time.sleep(5)
    
    # Open browser
    try:
        webbrowser.open("http://localhost:3000")
        print("🌐 Opening app in browser...")
    except:
        print("⚠️  Could not open browser automatically")
        print("   Please open http://localhost:3000 manually")
    
    print("\n🎉 App is running!")
    print("   Frontend: http://localhost:3000")
    print("   Backend API: http://localhost:8000")
    print("\n📋 Available API endpoints:")
    print("   GET  /health - Health check")
    print("   GET  /classes - List disease classes")
    print("   POST /predict - Analyze plant image")
    print("\n🛑 Press Ctrl+C to stop the application")
    
    try:
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
            
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
    
    # Cleanup
    print("🧹 Cleaning up processes...")
    if backend_process:
        backend_process.terminate()
    if frontend_process:
        frontend_process.terminate()
    
    print("✅ Application stopped successfully!")

if __name__ == "__main__":
    main()
