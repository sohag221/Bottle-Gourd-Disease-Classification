@echo off
title ngrok Authtoken Setup
color 0A

echo.
echo ======================================================
echo       🔑 ngrok Authtoken Setup
echo ======================================================
echo.

echo 📝 Step 1: Get your authtoken
echo    Go to: https://dashboard.ngrok.com/get-started/your-authtoken
echo    Copy your authtoken from the dashboard
echo.

set /p "token=🔑 Paste your authtoken here: "

if "%token%"=="" (
    echo ❌ No token provided!
    pause
    exit /b 1
)

echo.
echo 🔧 Setting up authtoken...
ngrok authtoken %token%

if %errorlevel% equ 0 (
    echo ✅ Authtoken set successfully!
    echo.
    echo 🚀 Now running the sharing app...
    echo.
    python share_app.py
) else (
    echo ❌ Failed to set authtoken
    echo Please check if the token is correct
)

echo.
pause
