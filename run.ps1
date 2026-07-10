# Start Backend
Write-Host "Starting FastAPI Backend on port 8000..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

# Start Frontend
Write-Host "Starting Streamlit Frontend on port 8501..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python -m streamlit run dashboard/app.py --server.port 8501"

Write-Host "Both servers are running!" -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000/docs"
Write-Host "Dashboard UI: http://localhost:8501"
Write-Host "Press Ctrl+C to stop (you may need to manually close the python processes in Task Manager)."
