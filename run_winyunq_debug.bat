@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"
cl /EHsc /std:c++17 WinyunqDebug/main.cpp /FeWinyunqDebug/WinyunqDebug.exe
if %ERRORLEVEL% NEQ 0 exit /b %ERRORLEVEL%
copy ThirdParty\LiteRtLm\Binaries\Win64\*.dll WinyunqDebug\ /y
cd WinyunqDebug
WinyunqDebug.exe
