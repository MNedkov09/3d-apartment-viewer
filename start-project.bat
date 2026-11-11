@echo off
echo ========================================================
echo   Starting 3D Apartment Viewer with AI Floorplan Detector
echo ========================================================

echo [1/2] Starting Python AI Detector backend on port 8000...
start "AI Detector Backend" cmd /k "cd /d %~dp0backend && .venv\Scripts\python.exe -m uvicorn app:app --port 8000 --host 127.0.0.1"

echo [2/2] Starting Next.js Frontend on port 3000...
start "Next.js Frontend" cmd /k "cd /d %~dp0 && npm run dev"

echo.
echo Both servers are launching!
echo App will be available at: http://localhost:3000
echo AI backend is at:         http://127.0.0.1:8000
echo.
pause
