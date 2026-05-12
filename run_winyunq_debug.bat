@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
cl /EHsc WinyunqDebug/main.cpp /FeWinyunqDebug/WinyunqDebug.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
copy bazel-bin\runtime\engine\litert_lm_wrapper.dll WinyunqDebug\ /y
copy bin\*.dll WinyunqDebug\ /y
copy prebuilt\windows_x86_64\*.dll WinyunqDebug\ /y
cd WinyunqDebug
WinyunqDebug.exe
