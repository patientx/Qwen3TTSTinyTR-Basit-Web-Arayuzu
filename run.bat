@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title Qwen3-TTS Tiny TR Basit Arabirim

set "PYTHONNOUSERSITE=1"
set "PY=%~dp0python_env\python.exe"

if not exist "%PY%" goto :install
"%PY%" -c "import qwen_tts, soundfile, numpy" >nul 2>&1
if errorlevel 1 goto :install
goto :start

:install
call "%~dp0install.bat" /auto
if errorlevel 1 goto :err

:start
"%PY%" "%~dp0server.py" %*
if errorlevel 1 pause
goto :eof

:err
echo Kurulum basarisiz oldu.
pause
exit /b 1
