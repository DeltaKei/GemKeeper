@echo off
echo Starting Chrome in Remote Debugging Mode (Port 9222)...
echo Please allow access if Windows Firewall asks.
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="C:\chrome_debug_profile"
pause
