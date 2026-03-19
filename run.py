#!/usr/bin/env python3
"""
Quick launcher for Graph RAG System
Runs both backend and frontend in separate processes
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_env_file():
    """Check if .env file exists"""
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  Warning: .env file not found!")
        print("Creating .env from .env_example...")
        
        example_path = Path(".env_example")
        if example_path.exists():
            with open(example_path, 'r') as src:
                content = src.read()
            with open(env_path, 'w') as dst:
                dst.write(content)
            print("✅ Created .env file")
            print("⚠️  Please add your OPENAI_API_KEY to .env file")
            return False
        else:
            print("❌ .env_example not found!")
            return False
    return True

def main():
    """Run both backend and frontend"""
    print("🔮 Graph RAG System Launcher")
    print("=" * 50)
    
    # Check environment
    if not check_env_file():
        print("\n⚠️  Please configure .env file and run again")
        return
    
    print("\n🚀 Starting Graph RAG System...")
    print("=" * 50)
    
    # Start backend
    print("\n📡 Starting Backend (http://localhost:8000)...")
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Wait for backend to start
    print("⏳ Waiting for backend to initialize...")
    time.sleep(3)
    
    # Start frontend
    print("\n🌐 Starting Frontend (http://localhost:7860)...")
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "app.frontend"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    print("\n" + "=" * 50)
    print("✅ Graph RAG System is running!")
    print("=" * 50)
    print("\n📍 Access points:")
    print("   • Frontend: http://localhost:7860")
    print("   • Backend API: http://localhost:8000")
    print("   • API Docs: http://localhost:8000/docs")
    print("\n⌨️  Press Ctrl+C to stop both services")
    print("=" * 50 + "\n")
    
    try:
        # Keep running and show output
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        backend_process.terminate()
        frontend_process.terminate()
        
        # Wait for processes to terminate
        backend_process.wait(timeout=5)
        frontend_process.wait(timeout=5)
        
        print("✅ Graph RAG System stopped")

if __name__ == "__main__":
    main()
