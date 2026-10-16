@echo off
setlocal
cd /d "%~dp0"
echo Procurement Pulse V2 - first-time setup
if exist ".venv\Scripts\python.exe" goto install
py -3.12 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" goto install
py -3.11 -m venv .venv >nul 2>nul
if exist ".venv\Scripts\python.exe" goto install
python -m venv .venv
if not exist ".venv\Scripts\python.exe" (
 echo Install Python 3.11 or 3.12 from python.org and select Add Python to PATH.
 pause
 exit /b 1
)
:install
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe run.py all
if errorlevel 1 goto failed
echo.
echo Setup complete. Double-click START_DASHBOARD.bat next.
pause
exit /b 0
:failed
echo Setup failed. Read the message above and the troubleshooting guide in README.md.
pause
exit /b 1
