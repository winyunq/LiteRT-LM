# Build the fork's current source into the official C DLL and wrapper DLL.
param(
    [string]$OutputDirectory = "ThirdParty/LiteRtLm",
    [string]$BazelOutputBase = "C:/b/output",
    [string]$BazelUserRoot = "C:/b/user"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$programFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
$vswhere = Join-Path $programFilesX86 "Microsoft Visual Studio/Installer/vswhere.exe"
$vsRoot = & $vswhere -latest -products "*" -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or !$vsRoot) { throw "MSVC installation not found" }
$env:BAZEL_VC = Join-Path $vsRoot "VC"
$env:BAZEL_SH = "C:/Program Files/Git/bin/bash.exe"
$env:ANDROID_HOME = $null
$env:ANDROID_SDK_ROOT = $null
$env:ANDROID_NDK_HOME = $null
$env:CARGO_BAZEL_REPIN = "true"

# Use the upstream Windows CI's long-path-aware linker setup.
$msvc = Get-ChildItem "$env:BAZEL_VC/Tools/MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1
$toolDir = Join-Path $msvc.FullName "bin/Hostx64/x64"
$mt = Get-ChildItem "$programFilesX86/Windows Kits/10/bin/*/x64/mt.exe" | Sort-Object FullName -Descending | Select-Object -First 1
$cacheRoot = Split-Path $BazelOutputBase -Parent
New-Item -ItemType Directory -Path $cacheRoot -Force | Out-Null
$manifest = Join-Path $cacheRoot "longpath.manifest"
[IO.File]::WriteAllText($manifest, '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0"><application xmlns="urn:schemas-microsoft-com:asm.v3"><windowsSettings><longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware></windowsSettings></application></assembly>')
& $mt.FullName -nologo -manifest $manifest "-outputresource:$toolDir/link.exe;1"
if ($LASTEXITCODE -ne 0) { throw "Could not enable linker long paths" }

$startup = @("--output_user_root=$BazelUserRoot", "--output_base=$BazelOutputBase")
$options = @(
    "--config=windows",
    "--features=static_link_msvc_runtimes",
    "--define=resolve_symbols_in_exec=false",
    "--repository_cache=$cacheRoot/repository-cache",
    "--disk_cache=$cacheRoot/disk-cache",
    "--repo_env=BAZEL_SH=$env:BAZEL_SH",
    "--repo_env=BAZEL_VC=$env:BAZEL_VC"
)
$targets = @("//c:litert-lm", "//runtime/engine:litert_lm_wrapper")
& bazel @startup build @options @targets
if ($LASTEXITCODE -ne 0) { throw "DLL build failed: $LASTEXITCODE" }

$execRoot = (& bazel @startup info execution_root | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not read Bazel execution root" }
$dllDependencies = & bazel @startup cquery 'deps(set(//c:litert-lm //runtime/engine:litert_lm_wrapper))' @options --output=files
if ($LASTEXITCODE -ne 0) { throw "Could not enumerate runtime DLLs" }

$bin = Join-Path $OutputDirectory "Binaries/Win64"
$include = Join-Path $OutputDirectory "Include"
New-Item -ItemType Directory -Path $bin, "$include/c", "$include/legacy/runtime/engine" -Force | Out-Null
foreach ($dll in @($dllDependencies | Where-Object { $_ -match '\.dll$' } | Sort-Object -Unique)) {
    Copy-Item -LiteralPath (Join-Path $execRoot $dll) -Destination $bin -Force
}
Copy-Item -LiteralPath "bazel-bin/c/litert-lm.dll", "bazel-bin/runtime/engine/litert_lm_wrapper.dll" -Destination $bin -Force
$coreImport = Get-ChildItem bazel-bin/c -Filter "litert-lm*.lib" | Select-Object -First 1
$wrapImport = Get-ChildItem bazel-bin/runtime/engine -Filter "litert_lm_wrapper*.lib" | Select-Object -First 1
Copy-Item -LiteralPath $coreImport.FullName -Destination "$bin/LiteRtLm_Native.lib" -Force
Copy-Item -LiteralPath $wrapImport.FullName -Destination "$bin/LiteRtLm_Extension.lib" -Force

foreach ($header in @("api_export.h", "engine.h", "conversation.h", "embedding_engine.h", "error_reporter.h", "experimental.h", "model_info.h")) {
    Copy-Item -LiteralPath "c/$header" -Destination "$include/c/$header" -Force
}
Copy-Item -LiteralPath "runtime/engine/litert_lm_api.h", "runtime/engine/litert_lm.hpp" -Destination $include -Force
Copy-Item -LiteralPath "runtime/engine/litert_lm_wrapper.h" -Destination "$include/legacy/runtime/engine/litert_lm_wrapper.h" -Force
Copy-Item -LiteralPath LICENSE -Destination $OutputDirectory -Force

$dumpbin = Join-Path $toolDir "dumpbin.exe"
$coreExports = & $dumpbin /EXPORTS "$bin/litert-lm.dll"
if ($LASTEXITCODE -ne 0 -or !($coreExports -match '\blitert_lm_get_api_version\b')) { throw "Core C API export missing" }
$wrapExports = & $dumpbin /EXPORTS "$bin/litert_lm_wrapper.dll"
if ($LASTEXITCODE -ne 0 -or !($wrapExports -match '\bLiteRtLm_GetApi\b')) { throw "Wrapper API export missing" }

$version = [IO.File]::ReadAllText("WRAPPER_VERSION").Trim()
$sourceVersion = [regex]::Match([IO.File]::ReadAllText("version.bzl"), '(?m)^VERSION = "([^"]+)"').Groups[1].Value
$cApiVersion = [regex]::Match([IO.File]::ReadAllText("version.bzl"), '(?m)^C_API_VERSION = "([^"]+)"').Groups[1].Value
$metadata = [ordered]@{
    wrapper_version = $version
    upstream_version = $sourceVersion
    upstream_commit = [IO.File]::ReadAllText("UPSTREAM_REVISION").Trim()
    source_commit = (git rev-parse HEAD).Trim()
    c_api_version = $cApiVersion
    build = "Windows x64 Release, MSVC static CRT"
    dll_sha256 = @{}
}
foreach ($dll in Get-ChildItem -LiteralPath $bin -Filter *.dll) {
    $metadata.dll_sha256[$dll.Name] = (Get-FileHash -LiteralPath $dll.FullName -Algorithm SHA256).Hash
}
[IO.File]::WriteAllText((Join-Path $OutputDirectory "version.json"), ($metadata | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
$archive = Join-Path (Split-Path $OutputDirectory -Parent) "LiteRtLm-$version-Win64.zip"
Compress-Archive -LiteralPath $OutputDirectory -DestinationPath $archive -Force
Write-Host "SDK: $archive"