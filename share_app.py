"""
Share Your Plant Disease Detection App Globally
This script helps you share your local Flask app with friends anywhere in the world
using ngrok tunneling service.
"""

import subprocess
import sys
import time
import requests
import json
from threading import Thread
import webbrowser

class AppSharer:
    def __init__(self):
        self.ngrok_process = None
        self.public_url = None
        
    def check_ngrok_installed(self):
        """Check if ngrok is installed"""
        try:
            result = subprocess.run(['ngrok', 'version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ ngrok is installed")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        print("❌ ngrok is not installed")
        return False
    
    def install_ngrok(self):
        """Guide user to install ngrok"""
        print("\n🔧 To share your app globally, you need to install ngrok:")
        print("\n📥 Option 1: Download and Install")
        print("1. Go to: https://ngrok.com/download")
        print("2. Download ngrok for Windows")
        print("3. Extract to a folder (e.g., C:\\ngrok\\)")
        print("4. Add the folder to your PATH environment variable")
        
        print("\n📦 Option 2: Using Chocolatey (if installed)")
        print("Run: choco install ngrok")
        
        print("\n📦 Option 3: Using winget")
        print("Run: winget install --id=ngrok.ngrok")
        
        print("\n🔑 After installation:")
        print("1. Sign up at https://ngrok.com (free account)")
        print("2. Get your auth token from dashboard")
        print("3. Run: ngrok authtoken YOUR_TOKEN")
        
    def get_ngrok_tunnels(self):
        """Get active ngrok tunnels"""
        try:
            response = requests.get('http://localhost:4040/api/tunnels')
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def start_ngrok_tunnel(self, port=5000):
        """Start ngrok tunnel for Flask app"""
        try:
            print(f"🚀 Starting ngrok tunnel for port {port}...")
            
            # Start ngrok in background
            self.ngrok_process = subprocess.Popen(
                ['ngrok', 'http', str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a bit for ngrok to start
            time.sleep(3)
            
            # Get tunnel URL from ngrok API
            tunnels = self.get_ngrok_tunnels()
            if tunnels and 'tunnels' in tunnels:
                for tunnel in tunnels['tunnels']:
                    if tunnel['proto'] == 'https':
                        self.public_url = tunnel['public_url']
                        print(f"✅ Public URL created: {self.public_url}")
                        return self.public_url
            
            print("⚠️ Could not get tunnel URL. Check ngrok status.")
            return None
            
        except Exception as e:
            print(f"❌ Error starting ngrok: {e}")
            return None
    
    def stop_ngrok(self):
        """Stop ngrok tunnel"""
        if self.ngrok_process:
            self.ngrok_process.terminate()
            print("🔴 ngrok tunnel stopped")
    
    def monitor_tunnel(self):
        """Monitor tunnel status"""
        print("\n🔍 Monitoring tunnel status...")
        print("📊 ngrok Web Interface: http://localhost:4040")
        print("💡 Press Ctrl+C to stop sharing")
        
        try:
            while True:
                time.sleep(30)
                tunnels = self.get_ngrok_tunnels()
                if tunnels and 'tunnels' in tunnels:
                    active_tunnels = len(tunnels['tunnels'])
                    print(f"✅ Tunnel active - {active_tunnels} connections")
                else:
                    print("⚠️ Tunnel may be disconnected")
        except KeyboardInterrupt:
            print("\n🔴 Stopping tunnel...")
            self.stop_ngrok()

def check_flask_app_running(port=5000):
    """Check if Flask app is running"""
    try:
        response = requests.get(f'http://localhost:{port}', timeout=5)
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def start_flask_app():
    """Start the Flask app"""
    import os
    os.chdir(r'd:\leaf_disease_app')
    
    # Use the virtual environment Python
    python_path = r'D:/leaf_disease_app/.venv/Scripts/python.exe'
    
    print("🚀 Starting Flask app...")
    flask_process = subprocess.Popen(
        [python_path, 'app.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for Flask to start
    print("⏳ Waiting for Flask app to start...")
    for i in range(30):  # Wait up to 30 seconds
        if check_flask_app_running():
            print("✅ Flask app is running!")
            return flask_process
        time.sleep(1)
        print(f"⏳ Waiting... ({i+1}/30)")
    
    print("❌ Flask app failed to start")
    return None

def main():
    print("🌿 Plant Disease Detection App - Global Sharing")
    print("=" * 50)
    
    sharer = AppSharer()
    
    # Check if ngrok is installed
    if not sharer.check_ngrok_installed():
        sharer.install_ngrok()
        input("\n⏸️ Press Enter after installing ngrok...")
        if not sharer.check_ngrok_installed():
            print("❌ Please install ngrok first")
            return
    
    # Check if Flask app is running
    if not check_flask_app_running():
        print("🔧 Flask app is not running. Starting it...")
        flask_process = start_flask_app()
        if not flask_process:
            print("❌ Could not start Flask app")
            return
    else:
        print("✅ Flask app is already running")
    
    # Start ngrok tunnel
    public_url = sharer.start_ngrok_tunnel()
    
    if public_url:
        print("\n🎉 SUCCESS! Your app is now accessible worldwide!")
        print("=" * 50)
        print(f"🌐 Share this URL with your friend: {public_url}")
        print("🔗 This URL works from anywhere in the world")
        print("📱 Your friend can open it on any device (phone, computer)")
        print("🔒 HTTPS secure connection")
        print("⏰ Free ngrok session lasts 2 hours")
        print("=" * 50)
        
        # Copy URL to clipboard (if possible)
        try:
            import pyperclip
            pyperclip.copy(public_url)
            print("📋 URL copied to clipboard!")
        except ImportError:
            print("💡 Install pyperclip to auto-copy URL: pip install pyperclip")
        
        # Open in browser
        try:
            webbrowser.open(public_url)
            print("🌐 Opening in your browser...")
        except:
            pass
        
        # Monitor tunnel
        sharer.monitor_tunnel()
    else:
        print("❌ Failed to create public tunnel")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

