@echo off
cd /d "%~dp0"
if exist runtime\pythonw.exe (
  start "" "%~dp0runtime\pythonw.exe" "%~dp0run.py"
  exit /b
)
if not exist .venv\Scripts\pythonw.exe (
  echo Python environment not found. See README.md for source installation.
  pause
  exit /b 1
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0run.py"
