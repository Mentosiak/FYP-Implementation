@echo off
setlocal

echo ============================================
echo Finalize Reports After Queue

echo ============================================

echo Running recovery comparisons...
call .\run_recovery_comparisons_queue.bat
if errorlevel 1 goto :fail

echo Regenerating benchmark summary...
.\.venv\Scripts\python.exe summarize_benchmarks.py
if errorlevel 1 goto :fail

echo Regenerating thesis report assets...
\.\.venv\Scripts\python.exe thesis_tools\generate_report_assets.py
if errorlevel 1 goto :fail

echo Finalization complete.
goto :end

:fail
echo Finalization failed.
exit /b 1

:end
endlocal
