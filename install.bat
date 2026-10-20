@echo off
setlocal
cd /d "%~dp0"
title Qwen3-TTS Tiny TR Basit Arabirim - Kurulum

rem Keep python_env isolated from user-level site-packages
set "PYTHONNOUSERSITE=1"
set "PY=%~dp0python_env\python.exe"

if exist "%PY%" goto :have_python

echo [*] [1/3] Python 3.12 indiriliyor...
if not exist "python_env" mkdir "python_env"
curl -L --ssl-no-revoke "https://www.python.org/ftp/python/3.12.9/python-3.12.9-embed-amd64.zip" -o "python_embed.zip" --no-progress-meter
if errorlevel 1 goto :failed

echo [*] [2/3] Python cikariliyor...
tar -xf "python_embed.zip" -C "python_env"
if errorlevel 1 goto :failed
del "python_embed.zip"

(
echo python312.zip
echo .
echo import site
) > "python_env\python312._pth"

:have_python
"%PY%" -m pip --version >nul 2>&1
if not errorlevel 1 goto :install_requirements

echo [*] [3/3] pip kuruluyor...
curl -L --ssl-no-revoke "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py" --no-progress-meter
if errorlevel 1 goto :failed
"%PY%" get-pip.py --no-warn-script-location
if errorlevel 1 goto :failed
del "get-pip.py"
"%PY%" -m pip install --upgrade pip setuptools==81 wheel --no-warn-script-location
if errorlevel 1 goto :failed

:install_requirements
echo [*] Gereksinimler kuruluyor...
"%PY%" -m pip install -r "%~dp0requirements.txt" --no-warn-script-location
if errorlevel 1 goto :failed

echo.
echo ====================================================
echo   Kurulum tamamlandi.
echo   Baslatmak icin run.bat dosyasini calistirin.
echo ====================================================
if /I not "%~1"=="/auto" pause
exit /b 0

:failed
echo.
echo ====================================================
echo   Kurulum basarisiz oldu. Yukaridaki hata mesajlarina bakin.
echo ====================================================
if /I not "%~1"=="/auto" pause
exit /b 1
