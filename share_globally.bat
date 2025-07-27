@echo off
title Plant Disease App - Global Sharing
color 0A

echo.
echo ======================================================
echo    🌿 Plant Disease Detection App - Global Sharing
echo ======================================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found!
    echo Please ensure you're in the correct directory
    pause
    exit /b 1
)

echo ✅ Virtual environment found
echo.

REM Install ngrok if needed
echo 🔧 Checking for ngrok...
ngrok version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ ngrok is not installed or not in PATH
    echo.
    echo 📥 To share globally, please install ngrok:
    echo    1. Go to: https://ngrok.com/download
    echo    2. Download ngrok for Windows
    echo    3. Extract to a folder (e.g., C:\ngrok\)
    echo    4. Add the folder to your PATH
    echo    5. Sign up at ngrok.com and get auth token
    echo    6. Run: ngrok authtoken YOUR_TOKEN
    echo.
    pause
    exit /b 1
)

echo ✅ ngrok is installed
echo.

REM Start Flask app in background
echo 🚀 Starting Flask app...
start /b .venv\Scripts\python.exe app.py

REM Wait for Flask to start
echo ⏳ Waiting for Flask app to start...
timeout /t 10 /nobreak >nul

REM Start ngrok tunnel
echo 🌐 Creating public tunnel...
echo.
echo 🎉 Starting ngrok tunnel for your app...
echo 💡 Your app will be accessible worldwide!
echo 📱 Share the HTTPS URL with your friend
echo ⏰ Free session lasts 2 hours
echo 🔴 Press Ctrl+C to stop sharing
echo.

ngrok http 5000

echo.
echo 👋 Tunnel stopped. Thank you for using the app!
pause
