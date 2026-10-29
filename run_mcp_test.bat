@echo off
echo ====================================================
echo   Winyunq: Running Native MCP Tester...
echo ====================================================

:: 0. Compile test program to ensure single-source alignment (步骤0: 物理编译)
echo [Compiling] Compiling mcp_native_test.cpp to align memory structures...
taskkill /f /im mcp_native_test.exe >nul 2>&1
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cl /EHsc /std:c++17 WinyunqDebug\mcp_native_test.cpp /FeWinyunqDebug\mcp_native_test.exe
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to compile mcp_native_test.cpp!
    exit /b 1
)

:: 1. Copy required dependency DLLs to WinyunqDebug directory (步骤2: 拼DLL)
echo [Deploying DLLs] Copying DLLs to WinyunqDebug...
taskkill /f /im mcp_native_test.exe >nul 2>&1

copy bazel-bin\runtime\engine\litert_lm_wrapper.dll WinyunqDebug\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to copy litert_lm_wrapper.dll to WinyunqDebug!
    exit /b 1
)

copy bin\dxcompiler.dll WinyunqDebug\ /y
if %ERRORLEVEL% NEQ 0 exit /b 1

copy bin\dxil.dll WinyunqDebug\ /y
if %ERRORLEVEL% NEQ 0 exit /b 1

copy bin\libLiteRt.dll WinyunqDebug\ /y
if %ERRORLEVEL% NEQ 0 exit /b 1

copy bin\libLiteRtWebGpuAccelerator.dll WinyunqDebug\ /y
if %ERRORLEVEL% NEQ 0 exit /b 1

copy bin\libLiteRtTopKWebGpuSampler.dll WinyunqDebug\ /y
if %ERRORLEVEL% NEQ 0 exit /b 1

copy bin\libGemmaModelConstraintProvider.dll WinyunqDebug\ /y
if %ERRORLEVEL% NEQ 0 exit /b 1

:: 2. One-click execute native verification (步骤2: 跑EXE)
cd WinyunqDebug
echo [Executing] mcp_native_test.exe
mcp_native_test.exe
echo ====================================================
echo   Tester execution finished.
echo ====================================================

