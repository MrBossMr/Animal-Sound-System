@echo off
setlocal enabledelayedexpansion
title Intelligent CDSS - Streamlit Server Launcher
color 0A

echo ===============================================================================
echo       INTELLIGENT CLINICAL DECISION SUPPORT SYSTEM (CDSS) - PRACTICAL 9
echo ===============================================================================
echo.

:: Step 1: Detect Python executable
set "PYTHON_EXE="

:: Check standard python in PATH
where python >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python"
    goto :PYTHON_FOUND
)

:: Check py launcher in PATH
where py >nul 2>nul
if %errorlevel% equ 0 (
    set "PYTHON_EXE=py"
    goto :PYTHON_FOUND
)

:: Search common User AppData locations
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
    if exist "%%D\python.exe" (
        set "PYTHON_EXE=%%D\python.exe"
        goto :PYTHON_FOUND
    )
)

:: Search Program Files
for /d %%D in ("%ProgramFiles%\Python*") do (
    if exist "%%D\python.exe" (
        set "PYTHON_EXE=%%D\python.exe"
        goto :PYTHON_FOUND
    )
)

:: Search Anaconda / Miniconda
if exist "%USERPROFILE%\anaconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\anaconda3\python.exe"
    goto :PYTHON_FOUND
)
if exist "%USERPROFILE%\miniconda3\python.exe" (
    set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
    goto :PYTHON_FOUND
)
if exist "C:\ProgramData\Anaconda3\python.exe" (
    set "PYTHON_EXE=C:\ProgramData\Anaconda3\python.exe"
    goto :PYTHON_FOUND
)

:PYTHON_NOT_FOUND
color 0C
echo [ERROR] Python was not detected on your system.
echo Please ensure Python 3.8+ is installed from https://www.python.org/
echo.
echo In the meantime, opening the standalone dashboard:
start "" "%~dp0dashboard.html"
echo.
pause
exit /b 1

:PYTHON_FOUND
echo [OK] Detected Python: !PYTHON_EXE!
echo.

:: Ensure in CDSS directory
cd /d "%~dp0"

:: Step 2: Verify / Install Streamlit and Core Libraries
echo [1/3] Checking dependencies...
!PYTHON_EXE! -m pip install streamlit numpy pandas matplotlib scikit-learn scipy soundfile
echo.

:: Step 3: Launch browser automatically after 3 seconds
echo [2/3] Launching your web browser to http://localhost:8501 ...
start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8501"

:: Step 4: Run Streamlit Server
echo [3/3] Starting Streamlit Server on port 8501...
echo.
echo ===============================================================================
echo  Application is running! Leave this command window open while using the app.
echo  To stop the server, press Ctrl + C in this window.
echo ===============================================================================
echo.

!PYTHON_EXE! -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false

pause
