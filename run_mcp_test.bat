@echo off
echo ====================================================
echo   Winyunq: Running Native MCP Tester...
echo ====================================================

:: 1. Copy required dependency DLLs to WinyunqDebug directory (步骤2: 拼DLL)
echo [Deploying DLLs] Copying DLLs to WinyunqDebug...
taskkill /f /im mcp_native_test.exe >nul 2>&1
copy ThirdParty\LiteRtLm\Binaries\Win64\*.dll WinyunqDebug\ /y

:: 2. One-click execute native verification (步骤2: 跑EXE)
cd WinyunqDebug
echo [Executing] mcp_native_test.exe
mcp_native_test.exe
cd ..
echo ====================================================
echo   Tester execution finished.
echo ====================================================

