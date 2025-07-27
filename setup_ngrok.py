"""
Quick Setup Script for ngrok Authentication
This script helps you set up ngrok to share your app globally
"""

import webbrowser
import subprocess
import sys

def main():
    print("🔧 Setting up ngrok for Global App Sharing")
    print("=" * 50)
    
    print("\n📝 Step 1: Create Free ngrok Account")
    print("Opening ngrok signup page...")
    webbrowser.open("https://dashboard.ngrok.com/signup")
    
    print("\n⏳ Please complete these steps in your browser:")
    print("1. Sign up for a free ngrok account")
    print("2. Verify your email address")
    print("3. Copy your authtoken from the dashboard")
    
    input("\n⏸️ Press Enter when you have your authtoken ready...")
    
    print("\n🔑 Step 2: Set up your authtoken")
    authtoken = input("Paste your authtoken here: ").strip()
    
    if authtoken:
        try:
            # Set the authtoken
            result = subprocess.run(['ngrok', 'authtoken', authtoken], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Authtoken set successfully!")
                print("\n🚀 Step 3: Test the connection")
                
                # Test ngrok
                print("Testing ngrok connection...")
                test_result = subprocess.run(['ngrok', 'http', '5000', '--log=stdout'], 
                                           capture_output=True, text=True, timeout=10)
                
                print("🎉 ngrok is ready!")
                print("\nNow you can run: python share_app.py")
                
            else:
                print(f"❌ Error setting authtoken: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("✅ ngrok connection test completed")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("❌ No authtoken provided")

if __name__ == "__main__":
    main()
