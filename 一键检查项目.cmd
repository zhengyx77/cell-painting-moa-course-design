@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0src\launcher.ps1" -Mode verify
pause
