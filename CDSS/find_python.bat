@echo off
echo =======================================================
echo          CHECKING PYTHON ENVIRONMENT STATUS
echo =======================================================
echo.

echo 1. Checking 'py' launcher...
where py
if %errorlevel% equ 0 (
    echo [FOUND py] Testing:
    py --version
) else (
    echo [NOT FOUND py]
)
echo.

echo 2. Checking 'python' in PATH...
where python
if %errorlevel% equ 0 (
    echo [FOUND python] Testing:
    python --version
) else (
    echo [NOT FOUND python in PATH]
)
echo.

echo 3. Checking AppData Python installations...
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python*") do (
    if exist "%%D\python.exe" (
        echo [FOUND in AppData]: %%D\python.exe
        "%%D\python.exe" --version
    )
)
echo.

echo 4. Checking Program Files Python installations...
for /d %%D in ("%ProgramFiles%\Python*") do (
    if exist "%%D\python.exe" (
        echo [FOUND in ProgramFiles]: %%D\python.exe
        "%%D\python.exe" --version
    )
)
echo.

pause
