# LiteRT-LM UE5 Wrapper 导出驱动器 (v2.1)
$ErrorActionPreference = "Stop"

echo "=========================================="
echo "LiteRT-LM UE5 Wrapper 导出 (Win64)"
echo "=========================================="

# 1. 环境准备
if (!(Test-Path "C:\bzl")) { New-Item -ItemType Directory -Path "C:\bzl" }
$env:CARGO_BAZEL_REPIN="true"

# 2. 编译 Wrapper DLL
echo "[1/3] 正在编译 litert_lm_wrapper.dll..."
bazel --output_base=C:\bzl build //runtime/engine:litert_lm_wrapper `
    --config=windows `
    --distdir=. `
    --define=litert_link_capi_so=true `
    --define=resolve_symbols_in_exec=false

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
