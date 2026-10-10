@echo off
rem GUI RVE500 — двойной клик. Использует .venv проекта, если есть.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" rve_gui.py
) else (
  python rve_gui.py
)
if errorlevel 1 pause
