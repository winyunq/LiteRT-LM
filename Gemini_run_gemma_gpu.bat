@echo off
setlocal

:: 设置模型路径
set MODEL_PATH=D:\gemma-4-E4B-it.litertlm

:: 进入 bin 目录，这里必须包含所有 GPU 相关的 .dll 文件
pushd bin

echo [System]: Starting Gemma 4 on GPU (RTX 4060)...
echo.

:: 直接启动程序。由于程序已经改成了交互式，它会自动等待你输入。
litert_lm_main.exe --backend=gpu --model_path="%MODEL_PATH%"

popd
pause
