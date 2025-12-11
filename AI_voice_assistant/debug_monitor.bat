@echo off
title Debug Log Monitor
if not exist "application.log" (
    echo Creating application.log...
    echo. > application.log
)
echo Tailing application.log...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-Content -Path 'application.log' -Wait -Tail 10"
pause
