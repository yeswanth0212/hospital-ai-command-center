@echo off
echo Starting Smart Hospital API Server...
echo The dashboard will be available at: http://localhost:8000/dashboard/
echo Press Ctrl+C to stop the server.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
