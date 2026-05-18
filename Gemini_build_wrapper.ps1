# LiteRT-LM UE5 Wrapper 导出驱动器 (v2.1)
$ErrorActionPreference = "Stop"

echo "=========================================="
echo "LiteRT-LM UE5 Wrapper 导出 (Win64)"
echo "=========================================="

# 1. 环境准备
if (!(Test-Path "C:\bzl")) { New-Item -ItemType Directory -Path "C:\bzl" }
# 强制指定 Git Bash 短路径，规避路径中包含空格的问题，并屏蔽未安装的 Android SDK 干扰
$env:BAZEL_SH = "C:/PROGRA~1/Git/bin/bash.exe"
$env:ANDROID_HOME = $null
$env:ANDROID_SDK_ROOT = $null
$env:ANDROID_NDK_HOME = $null
$env:CARGO_BAZEL_REPIN="true"

# 1.2 系统化集成：准备 SentencePiece 本地依赖 (绕过 WSL2 与网络波动)
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
            $content = $content -replace 'namespace util {', 'namespace util {'
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

# 2. 编译 Wrapper DLL
echo "[1/3] 正在编译 litert_lm_wrapper.dll (本地依赖覆盖模式)..."
bazel --output_base=C:\bzl build //runtime/engine:litert_lm_wrapper `
    --config=windows `
    --distdir=. `
    --define=litert_link_capi_so=true `
    --define=resolve_symbols_in_exec=false `
    --repo_env=BAZEL_SH=C:/PROGRA~1/Git/bin/bash.exe `
    --override_repository=sentencepiece=libs/sentencepiece

$env:CARGO_BAZEL_REPIN="false"

# 3. 整理发布到 ThirdParty
echo "[2/3] 整理发布目录 (D:\LiteRT-LM\ThirdParty\LiteRtLm)..."
$tpDir = "ThirdParty/LiteRtLm"
if (!(Test-Path "$tpDir/Include")) { New-Item -ItemType Directory -Path "$tpDir/Include" }
if (!(Test-Path "$tpDir/Binaries/Win64")) { New-Item -ItemType Directory -Path "$tpDir/Binaries/Win64" }

# 复制头文件
Copy-Item "runtime/engine/litert_lm_wrapper.h" "$tpDir/Include/" -Force

# 复制生成的 DLL 和 LIB
# 在 Windows 上，Bazel 可能会生成 .lib 或 .if.lib (interface library)
if (Test-Path "bazel-bin/runtime/engine/litert_lm_wrapper.lib") {
    Copy-Item "bazel-bin/runtime/engine/litert_lm_wrapper.lib" "$tpDir/Binaries/Win64/LiteRtLm_Wrapper.lib" -Force
} elseif (Test-Path "bazel-bin/runtime/engine/litert_lm_wrapper.if.lib") {
    Copy-Item "bazel-bin/runtime/engine/litert_lm_wrapper.if.lib" "$tpDir/Binaries/Win64/LiteRtLm_Wrapper.lib" -Force
} elseif (Test-Path "bazel-bin/runtime/engine/litert_lm_wrapper.dll.if.lib") {
    Copy-Item "bazel-bin/runtime/engine/litert_lm_wrapper.dll.if.lib" "$tpDir/Binaries/Win64/LiteRtLm_Wrapper.lib" -Force
} else {
    echo "[WARNING] Could not find import library (.lib). Ensure the DLL exports symbols."
}
Copy-Item "bazel-bin/runtime/engine/litert_lm_wrapper.dll" "$tpDir/Binaries/Win64/" -Force

# 4. 搜集并复制所有依赖的运行时 DLLs
echo "[3/3] 补全运行时依赖 (libLiteRt.dll, dxcompiler.dll 等)..."
# 注意：这些 DLL 应该已经在你的 bin 目录或者 prebuilt 目录中
$dependencies = @(
    "libLiteRt.dll",
    "libLiteRtWebGpuAccelerator.dll",
    "libLiteRtTopKWebGpuSampler.dll",
    "libGemmaModelConstraintProvider.dll",
    "dxcompiler.dll",
    "dxil.dll"
)

foreach ($dll in $dependencies) {
    if (Test-Path "bin/$dll") {
        Copy-Item "bin/$dll" "$tpDir/Binaries/Win64/" -Force
    } elseif (Test-Path "prebuilt/windows_x86_64/$dll") {
        Copy-Item "prebuilt/windows_x86_64/$dll" "$tpDir/Binaries/Win64/" -Force
    }
}

echo "=========================================="
echo "Wrapper 导出成功！"
echo "目录: $tpDir"
echo "=========================================="
