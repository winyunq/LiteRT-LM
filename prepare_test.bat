@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
echo [1/2] Compiling WinyunqDebug.exe...
cl /EHsc WinyunqDebug/main.cpp /FeWinyunqDebug/WinyunqDebug.exe
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Compilation failed.
    exit /b %ERRORLEVEL%
)

echo [2/2] Gathering Runtime DLLs...
copy bazel-bin\runtime\engine\litert_lm_wrapper.dll WinyunqDebug\ /y
copy bin\dxcompiler.dll WinyunqDebug\ /y
copy bin\dxil.dll WinyunqDebug\ /y
copy bin\libLiteRt.dll WinyunqDebug\ /y
copy bin\libLiteRtWebGpuAccelerator.dll WinyunqDebug\ /y
copy bin\libLiteRtTopKWebGpuSampler.dll WinyunqDebug\ /y
copy bin\libGemmaModelConstraintProvider.dll WinyunqDebug\ /y

echo.
echo =======================================================
echo  READY: D:\LiteRT-LM\WinyunqDebug\WinyunqDebug.exe
echo =======================================================
echo Please run the exe to test the full GPU conversation.
