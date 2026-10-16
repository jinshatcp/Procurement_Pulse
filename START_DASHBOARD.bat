@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
 echo First run SETUP_WINDOWS.bat.
 pause
 exit /b 1
)
if not exist "outputs\model_bundle.joblib" (
 echo Training the demonstration models...
 .venv\Scripts\python.exe run.py all
 if errorlevel 1 (pause & exit /b 1)
)
.venv\Scripts\python.exe -m streamlit run dashboard.py
pause
