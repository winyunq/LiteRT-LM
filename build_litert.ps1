# LiteRT-LM 纯净编译驱动器
$ErrorActionPreference = "Stop"

echo "=========================================="
echo "LiteRT-LM 纯净编译启动 (13900HX + RTX 4060)"
echo "=========================================="

# 1. 清理
echo "[1/4] 清理旧环境..."
bazel shutdown
if (Test-Path "bin") { Remove-Item -Recurse -Force "bin" -ErrorAction SilentlyContinue }

# 1.1 系统化集成：处理 SentencePiece 源码 (绕过损坏的 WSL2)
if (!(Test-Path "libs")) { New-Item -ItemType Directory -Path "libs" }
$spDir = "libs/sentencepiece"
if (!(Test-Path $spDir)) {
    echo "正在准备 SentencePiece 源码..."
    $spUrl = "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz"
    $spZip = "libs/sp.tar.gz"
    Invoke-WebRequest -Uri $spUrl -OutFile $spZip
    tar -xf $spZip -C libs/
    Move-Item "libs/sentencepiece-0.2.0" $spDir
    Remove-Item $spZip
    
    echo "系统化修复源码路径与兼容性..."
    # 扩大搜索范围到根目录，以处理 third_party 下的头文件
    Get-ChildItem -Path "$spDir" -Include *.h,*.cc,*.proto,*.hxx -Recurse | ForEach-Object {
        $content = Get-Content $_.FullName -Raw
        $content = $content -replace 'third_party/absl/', 'absl/'
        $content = $content -replace 'third_party/protobuf-lite/google/protobuf/', 'google/protobuf/'
        
        if ($_.Name -eq "common.h") {
            $content = "#include <cstdlib>`n" + $content
            $content = $content -replace 'void Abort\(\);', ''
            $content = $content -replace 'Abort\(\);', '::abort();'
        }
        
        if ($_.Name -eq "error.cc") {
            "// Removed to use external absl::Status" | Set-Content $_.FullName
        }
        
        if ($_.Name -eq "sentencepiece_processor.h") {
            $content = $content -replace 'namespace util {', "namespace util {`nusing StatusCode = absl::StatusCode;`nusing Status = absl::Status;"
            $content = $content -replace '(?s)enum class StatusCode : int \{.*?\};', ''
            $content = $content -replace '(?s)class Status \{.*?\};', ''
            $content = $content -replace '#include <vector>', "#include <vector>`n#include `"absl/status/status.h`""
            $content | Set-Content $_.FullName
        }
        
        if ($_.Name -ne "error.cc" -and $_.Name -ne "sentencepiece_processor.h") {
            $content | Set-Content $_.FullName
        }
    }
}

# 复制 BUILD 文件并创建空的 WORKSPACE 文件，以作为本地 Override 库运行
Copy-Item "BUILD.sentencepiece" "$spDir/BUILD.bazel" -Force
if (!(Test-Path "$spDir/WORKSPACE")) {
    New-Item -ItemType File -Path "$spDir/WORKSPACE" -Force | Out-Null
}

# 2. 编译
echo "[2/4] 正在调用 Bazel 编译 (本地依赖覆盖模式 + GPU 优化)..."
if (!(Test-Path "C:\bzl")) { New-Item -ItemType Directory -Path "C:\bzl" }
# 强制指定 Git Bash 短路径，规避路径中包含空格的问题，并屏蔽未安装 of Android SDK 干扰
$env:BAZEL_SH = "C:/PROGRA~1/Git/bin/bash.exe"
$env:ANDROID_HOME = $null
$env:ANDROID_SDK_ROOT = $null
$env:ANDROID_NDK_HOME = $null

# 强制使用本地压缩包，规避 WSL2/sed 依赖
# 增加 CARGO_BAZEL_REPIN=true 解决 Rust 依赖指纹变更问题
# 根据文档添加 GPU 支持所需的定义：
# --define=litert_link_capi_so=true 链接 C API 共享库
# --define=resolve_symbols_in_exec=false 解决 GPU 驱动符号解析问题
$env:CARGO_BAZEL_REPIN="true"
bazel --output_base=C:\bzl build //runtime/engine:litert_lm_main `
    --config=windows `
    --distdir=. `
    --define=litert_link_capi_so=true `
    --define=resolve_symbols_in_exec=false `
    --repo_env=BAZEL_SH=C:/PROGRA~1/Git/bin/bash.exe `
    --override_repository=sentencepiece=libs/sentencepiece
$env:CARGO_BAZEL_REPIN="false"

# 3. 发布
echo "[3/4] 整理发布目录..."
if (!(Test-Path "bin")) { New-Item -ItemType Directory -Path "bin" }
Copy-Item "bazel-bin/runtime/engine/litert_lm_main.exe" "bin/" -Force
Copy-Item "prebuilt/windows_x86_64/*.dll" "bin/" -Force

# 搜集 GPU 依赖 (DirectX Shader Compiler)
# 优先从 Windows SDK 路径获取最新的编译器，如果没有则检查 prebuilt
$sdkPath = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64"
if (Test-Path "$sdkPath\dxcompiler.dll") {
    Copy-Item "$sdkPath\dxil.dll" "bin/" -Force
    Copy-Item "$sdkPath\dxcompiler.dll" "bin/" -Force
    echo "GPU 编译器已从 Windows SDK 同步。"
} elseif (Test-Path "prebuilt/windows_x86_64/dxcompiler.dll") {
    Copy-Item "prebuilt/windows_x86_64/dxil.dll" "bin/" -Force
    Copy-Item "prebuilt/windows_x86_64/dxcompiler.dll" "bin/" -Force
    echo "GPU 编译器已从 prebuilt 目录同步。"
}

echo "=========================================="
echo "编译成功！"
echo "=========================================="
