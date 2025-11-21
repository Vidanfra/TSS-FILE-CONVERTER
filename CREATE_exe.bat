@echo off
echo Checking if PyInstaller is installed...
pyinstaller --version >nul 2>&1
if %errorlevel% neq 0 (
    echo PyInstaller is not installed or not in the PATH.
    echo Please install it using: pip install pyinstaller
    pause
    exit /b
)

echo PyInstaller is installed.
echo.
echo Starting the build process for TSSconverter4...
echo This may take a few minutes.
echo.

pyinstaller --noconsole --onefile --clean --name "TSSconverter4" --exclude-module PyQt5 --exclude-module PyQt6 --exclude-module PySide2 --exclude-module PySide6 --exclude-module notebook --exclude-module share --exclude-module scipy TSSconverter4.py

echo.
echo Build process completed!
echo You can find the executable in the 'dist' folder.
pause