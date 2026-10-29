@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
cl /EHsc /std:c++17 WinyunqDebug/main.cpp /FeWinyunqDebug/WinyunqDebug.exe
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Compilation failed!
    exit /b %ERRORLEVEL%
)

echo [Sync] Copying latest generated DLLs directly to WinyunqDebug...
taskkill /f /im WinyunqDebug.exe >nul 2>&1

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

cd WinyunqDebug
WinyunqDebug.exe
