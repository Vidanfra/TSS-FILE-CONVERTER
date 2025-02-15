REM Run this script to create an executable file from the python script

REM Check if pyinstaller is installed
pyinstaller --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo "PyInstaller is not installed. Please install it using 'pip install pyinstaller'."
    exit /b 1
)

echo "Creating executable file from python script..."
pyinstaller --onefile TSSconverter3.py
echo "Executable file created successfully!"