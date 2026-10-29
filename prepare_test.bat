@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
echo [1/2] Compiling WinyunqDebug.exe...
cl /EHsc /std:c++17 WinyunqDebug/main.cpp /FeWinyunqDebug/bin/WinyunqDebug.exe
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Compilation failed!
    exit /b %ERRORLEVEL%
)

echo [2/2] Gathering Runtime DLLs to WinyunqDebug\bin...
:: 强杀可能占用的测试进程，防止 DLL 被占用锁定
taskkill /f /im WinyunqDebug.exe >nul 2>&1
taskkill /f /im UnrealEditor.exe >nul 2>&1

:: 从 ThirdParty 物理目录安全同步，彻底避开 bazel-bin 在非管理员终端下的符号链接读取拦截
echo [Sync] Copying DLLs from ThirdParty to WinyunqDebug\bin...
if exist WinyunqDebug\bin\litert_lm_wrapper.dll del /f /q WinyunqDebug\bin\litert_lm_wrapper.dll >nul 2>&1

copy ThirdParty\LiteRtLm\Binaries\Win64\litert_lm_wrapper.dll WinyunqDebug\bin\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to sync litert_lm_wrapper.dll to bin!
    exit /b 1
)

copy ThirdParty\LiteRtLm\Binaries\Win64\dxcompiler.dll WinyunqDebug\bin\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to sync dxcompiler.dll to bin!
    exit /b 1
)

copy ThirdParty\LiteRtLm\Binaries\Win64\dxil.dll WinyunqDebug\bin\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to sync dxil.dll to bin!
    exit /b 1
)

copy ThirdParty\LiteRtLm\Binaries\Win64\libLiteRt.dll WinyunqDebug\bin\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to sync libLiteRt.dll to bin!
    exit /b 1
)

copy ThirdParty\LiteRtLm\Binaries\Win64\libLiteRtWebGpuAccelerator.dll WinyunqDebug\bin\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to sync libLiteRtWebGpuAccelerator.dll to bin!
    exit /b 1
)

copy ThirdParty\LiteRtLm\Binaries\Win64\libLiteRtTopKWebGpuSampler.dll WinyunqDebug\bin\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to sync libLiteRtTopKWebGpuSampler.dll to bin!
    exit /b 1
)

copy ThirdParty\LiteRtLm\Binaries\Win64\libGemmaModelConstraintProvider.dll WinyunqDebug\bin\ /y
if %ERRORLEVEL% NEQ 0 (
    echo [FATAL ERROR] Failed to sync libGemmaModelConstraintProvider.dll to bin!
    exit /b 1
)



echo.
echo =======================================================
echo  READY: D:\LiteRT-LM\WinyunqDebug\WinyunqDebug.exe
echo =======================================================
echo Please run the exe to test the full GPU conversation.
