@echo off
REM ============================================================
REM EIVA WorkFlow Manager - Command Line Execution Script
REM ============================================================
REM This script runs the WFM_import_and_export.xml workflow
REM from the command line without needing the GUI.
REM ============================================================

REM Set paths
set WFM_EXE=C:\Eiva\WorkFlowManager\WorkflowManager.exe
set SCRIPT_PATH=C:\Users\Administrator\Documents\Vicente\TSS-FILE-CONVERTER\WFM\WFM_import_and_export.xml

echo ============================================================
echo Running EIVA WorkFlow Manager Script
echo ============================================================
echo.
echo WFM Executable: %WFM_EXE%
echo Script: %SCRIPT_PATH%
echo.
echo Starting workflow...
echo.

REM Run WorkFlowManager with the XML script as argument
"%WFM_EXE%" "%SCRIPT_PATH%"

echo.
echo ============================================================
echo Workflow execution completed.
echo ============================================================
pause
