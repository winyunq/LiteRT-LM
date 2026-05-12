@echo off
setlocal enabledelayedexpansion

:: 设置模型路径
set MODEL_PATH=D:\gemma-4-E4B-it.litertlm

if not exist "%MODEL_PATH%" (
    echo [ERROR] Model file not found: %MODEL_PATH%
    pause
    exit /b
)

:: 进入 bin 目录
pushd bin

echo =======================================================
echo  LiteRT-LM Gemma 4 Interactive CLI (CPU Mode)
echo =======================================================
echo.
echo [System]: CPU Multi-threading is enabled (13900HX).
echo [System]: Enter your prompt below. Type 'exit' to quit.
echo.

:CHAT_LOOP
echo -------------------------------------------------------
:: 获取用户输入
set "USER_PROMPT="
set /p "USER_PROMPT=User >> "

:: 检查是否想要退出
if /i "%USER_PROMPT%"=="exit" goto CHAT_END
if "%USER_PROMPT%"=="" goto CHAT_LOOP

echo.
echo [Gemma 4 is thinking...]
echo.

:: 运行推理程序
litert_lm_main.exe --backend=cpu --model_path="%MODEL_PATH%" --input_prompt="%USER_PROMPT%"

echo.
:: 返回循环
goto CHAT_LOOP

:CHAT_END
echo.
echo Goodbye!
popd
pause
