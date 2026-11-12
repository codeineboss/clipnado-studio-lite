@echo off
cd /d "%~dp0"
if exist runtime\python.exe (
  "%~dp0runtime\python.exe" "%~dp0scripts\diagnostics.py"
) else (
  .venv\Scripts\python.exe scripts\diagnostics.py
)
pause
