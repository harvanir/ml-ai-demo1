@echo off
echo Stopping ML-AI-Demo1 application...
taskkill /IM python.exe /F >nul 2>&1
if %errorlevel% equ 0 (
    echo Application stopped successfully.
) else (
    echo No Python processes found or failed to stop.
)
pause