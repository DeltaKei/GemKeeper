@echo off
echo.
echo ==========================================================
echo   [1/2] Starting Chrome in Remote Debugging Mode (Port 9222)...
echo ==========================================================

:: Chrome을 새 창으로 실행하고 (start "") 명령어를 사용하여 대기하지 않고 다음 줄로 넘어갑니다.
:: run_chrome.bat의 전체 경로를 그대로 사용합니다. 
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug_profile"

echo.
echo [INFO] Waiting 5 seconds for Chrome to fully start and initialize CDP...
:: 5초간 대기하여 Chrome이 원격 디버깅 모드를 완전히 준비할 시간을 줍니다.
timeout /t 5 /nobreak > nul

echo.
echo ==========================================================
echo   [2/2] Starting GemKeeper Python Script (Compact ^& Silent)...
echo ==========================================================

:: [핵심] pythonw.exe를 사용하여 콘솔 창(터미널) 없이 백그라운드에서 실행합니다.
:: 에러가 발생해도 터미널이 뜨지 않으므로, 오류 감지는 Watchdog에 의존해야 합니다.
pythonw.exe "%~dp0gem_keeper.py"

echo.
echo [SUCCESS] GemKeeper Initialization Complete.
echo.
echo **Python 스크립트는 백그라운드에서 실행 중입니다. 창을 닫아도 계속 작동합니다.**
pause