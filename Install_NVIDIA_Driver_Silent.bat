@echo off
:: Check for Administrator privileges and self-elevate
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo ========================================================
echo   Installing NVIDIA 597.06 Production Driver (Silent)
echo ========================================================
echo Installing in background with ZERO clicks required...
echo Please wait about 60-90 seconds...
echo.

"C:\Users\bharg\Downloads\NVIDIA_597.06\setup.exe" -s -noreboot

echo.
echo Installing hardware driver INF via Windows PnP...
pnputil /add-driver "C:\Users\bharg\Downloads\NVIDIA_597.06\Display.Driver\nvdmwi.inf" /install

echo.
echo ========================================================
echo   Installation Complete!
echo ========================================================
timeout /t 5
