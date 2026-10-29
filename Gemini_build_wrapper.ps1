# LiteRT-LM native SDK export driver (v4.2 / upstream v0.16.1 + stable ABI 1.4)
$ErrorActionPreference = "Stop"

echo "=========================================="
echo "LiteRT-LM UE5 Wrapper 导出 (Win64)"
echo "=========================================="

# 1. 环境准备
if (!(Test-Path "D:\LiteRTLMBuildCache")) { New-Item -ItemType Directory -Path "D:\LiteRTLMBuildCache" }
# 强制指定 Git Bash 短路径，规避路径中包含空格的问题，并屏蔽未安装的 Android SDK 干扰
$env:BAZEL_SH = "C:/PROGRA~1/Git/bin/bash.exe"
$env:BAZEL_VC = "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC"
$env:PATH = "C:\Program Files\Git\bin;" + $env:PATH
$env:ANDROID_HOME = $null
$env:ANDROID_SDK_ROOT = $null
$env:ANDROID_NDK_HOME = $null
# rules_rust treats every non-empty CARGO_BAZEL_REPIN value as a package
# selector; the literal string "false" therefore tries to repin a package
# named false.  A release build must consume the checked-in lockfile.
Remove-Item Env:CARGO_BAZEL_REPIN -ErrorAction SilentlyContinue
$BazelOutputBase = "D:\bzl_a"

function Assert-Sha256([string]$Path, [string]$Expected) {
    if (!(Test-Path $Path)) { throw "Required artifact is missing: $Path" }
    $Actual = (Get-FileHash $Path -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($Actual -ne $Expected.ToUpperInvariant()) {
        throw "SHA256 mismatch for $Path. Expected $Expected, got $Actual"
    }
}

# The Windows wheel is the official v0.16.1 C core used by LiteRT-LM's own
# Python distribution. It contains the statically linked WebGPU accelerator;
# locally rebuilding //c:litert-lm instead selects the incompatible dynamic
# accelerator bundle on Windows.
$NativeVersion = "0.16.1"
$WheelUrl = "https://files.pythonhosted.org/packages/31/42/43ffd35663f035bb3fe722c71a576b9fb482c2b97b1a697aba2c727cf8ff/litert_lm_api-0.16.1-py3-none-win_amd64.whl"
$WheelSha256 = "B45D27A46D4EF048E1F29AE514B979BFB92C69710776A1F86456367FFAE5ACBC"
$CoreSha256 = "DE7DA4352C522368E44418996A8B2942B44BFC647912E79A03B0D3A9D59683EC"
$DxcompilerSha256 = "B86A738ECE4C05DBE2D9BBB29668A2CCB28A0740773C9B027CDC61E8D07B4D75"
$DxilSha256 = "058F2F52A680C38B223A5615B7DF969DEF21F77E577562E5099D971DEDE992DE"
$WheelPath = "D:\LiteRTLMBuildCache\litert_lm_api-$NativeVersion-py3-none-win_amd64.whl"
$NativeExtractDir = "D:\LiteRTLMBuildCache\litert_lm_api-$NativeVersion-win_amd64"

echo "[1/4] 准备并校验官方 LiteRT-LM $NativeVersion Windows C 核心..."
if (!(Test-Path $WheelPath)) {
    Invoke-WebRequest -Uri $WheelUrl -OutFile $WheelPath
}
Assert-Sha256 $WheelPath $WheelSha256
if (!(Test-Path "$NativeExtractDir\litert_lm\litert-lm.dll")) {
    New-Item -ItemType Directory -Path $NativeExtractDir -Force | Out-Null
    tar -xf $WheelPath -C $NativeExtractDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to extract official LiteRT-LM wheel"
    }
}
$OfficialDir = "$NativeExtractDir\litert_lm"
Assert-Sha256 "$OfficialDir\litert-lm.dll" $CoreSha256
Assert-Sha256 "$OfficialDir\dxcompiler.dll" $DxcompilerSha256
Assert-Sha256 "$OfficialDir\dxil.dll" $DxilSha256

# 2. Compile only the small extension bridge. The model/runtime implementation
# remains in the official sibling DLL and its complete C ABI stays callable.
echo "[2/4] 正在编译稳定扩展桥接 DLL..."
bazel --output_base=$BazelOutputBase build //runtime/engine:litert_lm_native_bridge `
    --config=windows `
    --distdir=. `
    --features=static_link_msvc_runtimes `
    --define=resolve_symbols_in_exec=false `
    --repo_env=BAZEL_SH=C:/PROGRA~1/Git/bin/bash.exe `
    --repo_env=BAZEL_VC="C:\Program Files\Microsoft Visual Studio\18\Insiders\VC"

if ($LASTEXITCODE -ne 0) {
    throw "litert_lm_native_bridge.dll build failed with exit code $LASTEXITCODE"
}

# 3. 整理发布到 ThirdParty
echo "[3/4] 整理完整原生 SDK (D:\LiteRT-LM\ThirdParty\LiteRtLm)..."
$tpDir = "ThirdParty/LiteRtLm"
if (!(Test-Path "$tpDir/Include")) { New-Item -ItemType Directory -Path "$tpDir/Include" }
if (!(Test-Path "$tpDir/Binaries/Win64")) { New-Item -ItemType Directory -Path "$tpDir/Binaries/Win64" }
if (!(Test-Path "$tpDir/Include/runtime/engine")) { New-Item -ItemType Directory -Path "$tpDir/Include/runtime/engine" -Force | Out-Null }
if (!(Test-Path "$tpDir/Include/c")) { New-Item -ItemType Directory -Path "$tpDir/Include/c" -Force | Out-Null }

# Canonical SDK headers: the complete upstream C API, its no-loss C++ RAII
# facade, and the optional stable extension ABI used by Unreal.  The old
# CamelCase source wrapper remains available below Include/legacy, but it must
# not be mistaken for the complete/native API or linked against the bridge.
$headers = @(
    "litert_lm_api.h",
    "litert_lm.hpp",
    "litert_lm_native_api.h",
    "litert_lm_native.hpp"
)
foreach ($header in $headers) {
    Copy-Item "runtime/engine/$header" "$tpDir/Include/$header" -Force
    Copy-Item "runtime/engine/$header" "$tpDir/Include/runtime/engine/$header" -Force
}
Copy-Item "c/engine.h" "$tpDir/Include/c/engine.h" -Force

$LegacyIncludeDir = "$tpDir/Include/legacy/runtime/engine"
New-Item -ItemType Directory -Path $LegacyIncludeDir -Force | Out-Null
Copy-Item "runtime/engine/litert_lm_wrapper.h" `
    "$LegacyIncludeDir/litert_lm_wrapper.h" -Force

# Exact obsolete release paths from the former source-built/dynamic-WebGPU
# bundle.  Their source targets are retained in the repository; removing them
# here prevents a consumer from accidentally loading an ABI-incompatible mix.
$ObsoleteReleaseFiles = @(
    "$tpDir/Include/litert_lm_wrapper.h",
    "$tpDir/Include/runtime/engine/litert_lm_wrapper.h",
    "$tpDir/Binaries/Win64/LiteRtLm_Wrapper.lib",
    "$tpDir/Binaries/Win64/libGemmaModelConstraintProvider.dll",
    "$tpDir/Binaries/Win64/libLiteRt.dll",
    "$tpDir/Binaries/Win64/libLiteRtTopKWebGpuSampler.dll",
    "$tpDir/Binaries/Win64/libLiteRtWebGpuAccelerator.dll",
    "$tpDir/Binaries/Win64/libwebgpu_dawn.dll"
)
foreach ($ObsoleteReleaseFile in $ObsoleteReleaseFiles) {
    if (Test-Path $ObsoleteReleaseFile) {
        Remove-Item -LiteralPath $ObsoleteReleaseFile -Force
    }
}

# Copy the extension under the stable deployment name.
$BridgeOutput = "bazel-bin/runtime/engine/litert_lm_native_bridge.dll"
Copy-Item $BridgeOutput "$tpDir/Binaries/Win64/litert_lm_wrapper.dll" -Force
$BridgeImportCandidates = @(
    "bazel-bin/runtime/engine/litert_lm_native_bridge.if.lib",
    "bazel-bin/runtime/engine/litert_lm_native_bridge.lib",
    "bazel-bin/runtime/engine/litert_lm_native_bridge.dll.if.lib"
)
$BridgeImport = $BridgeImportCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($BridgeImport) {
    Copy-Item $BridgeImport "$tpDir/Binaries/Win64/LiteRtLm_Extension.lib" -Force
} else {
    throw "The extension bridge import library was not produced"
}

Copy-Item "$OfficialDir\litert-lm.dll" "$tpDir/Binaries/Win64/litert-lm.dll" -Force
Copy-Item "$OfficialDir\dxcompiler.dll" "$tpDir/Binaries/Win64/dxcompiler.dll" -Force
Copy-Item "$OfficialDir\dxil.dll" "$tpDir/Binaries/Win64/dxil.dll" -Force

# Generate a normal MSVC import library for every upstream litert_lm_* export.
# C++ consumers can link this, while dynamic consumers can ignore it and load
# the same complete official DLL directly.
$MsvcTools = Get-ChildItem "$env:BAZEL_VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1
$Dumpbin = "$($MsvcTools.FullName)\bin\Hostx64\x64\dumpbin.exe"
$LibTool = "$($MsvcTools.FullName)\bin\Hostx64\x64\lib.exe"
$NativeDll = "$tpDir/Binaries/Win64/litert-lm.dll"
$NativeDef = "$tpDir/Binaries/Win64/litert-lm.def"
$NativeLib = "$tpDir/Binaries/Win64/LiteRtLm_Native.lib"
$NativeExp = "$tpDir/Binaries/Win64/LiteRtLm_Native.exp"
$ExportNames = @(& $Dumpbin /EXPORTS $NativeDll | ForEach-Object {
    if ($_ -match '^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]+\s+(litert_lm_[A-Za-z0-9_]+)\s*$') { $Matches[1] }
}) | Sort-Object -Unique
$HeaderText = Get-Content "c/engine.h" -Raw
$DeclaredNames = @([regex]::Matches($HeaderText, '\b(litert_lm_[a-zA-Z0-9_]+)\s*\(') | ForEach-Object { $_.Groups[1].Value }) | Sort-Object -Unique
$MissingExports = @($DeclaredNames | Where-Object { $_ -notin $ExportNames })
if ($MissingExports.Count -ne 0) {
    throw "Official core does not export declarations from c/engine.h: $($MissingExports -join ', ')"
}
@("LIBRARY litert-lm.dll", "EXPORTS") + @($ExportNames | ForEach-Object { "  $_" }) | Set-Content $NativeDef -Encoding Ascii
& $LibTool /nologo "/def:$NativeDef" /machine:x64 "/out:$NativeLib"
if ($LASTEXITCODE -ne 0) { throw "Failed to generate LiteRtLm_Native.lib" }
if (Test-Path $NativeExp) {
    Remove-Item -LiteralPath $NativeExp -Force
}

echo "[4/4] 校验发布哈希和完整官方导出表..."
Assert-Sha256 $NativeDll $CoreSha256
Assert-Sha256 "$tpDir/Binaries/Win64/dxcompiler.dll" $DxcompilerSha256
Assert-Sha256 "$tpDir/Binaries/Win64/dxil.dll" $DxilSha256
if ($ExportNames.Count -lt 100) {
    throw "Unexpectedly small official C API export set: $($ExportNames.Count)"
}

echo "=========================================="
echo "Native SDK 导出成功：官方 C API $($ExportNames.Count) 个符号 + 稳定扩展 ABI 1.4"
echo "目录: $tpDir"
echo "=========================================="
