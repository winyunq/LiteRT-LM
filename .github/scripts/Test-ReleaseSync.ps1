$ErrorActionPreference = 'Stop'
$syncScript = Join-Path $PSScriptRoot 'Sync-UpstreamRelease.ps1'
$sandbox = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
[IO.Directory]::CreateDirectory((Join-Path $sandbox '.github')) | Out-Null
[IO.Directory]::CreateDirectory((Join-Path $sandbox 'runtime/engine')) | Out-Null
[IO.File]::WriteAllText((Join-Path $sandbox '.github/upstream-release-baseline.txt'), 'v0.18.1')
[IO.File]::WriteAllText((Join-Path $sandbox 'UPSTREAM_REVISION'), 'baseline-sha')
$calls = [Collections.Generic.List[string]]::new()
function git {
    $global:LASTEXITCODE = 0
    $call = $args -join ' '
    $calls.Add("git $call")
    switch -Wildcard ($call) {
        'for-each-ref *' { 'upstream-sha refs/google/tags/v0.20.0' }
        'rev-parse HEAD' { 'fork-sha' }
        'rev-parse refs/google/tags/*' { 'upstream-sha' }
        'lfs ls-files *' { 'lfs-oid * prebuilt/windows_x86_64/runtime.dll' }
    }
}
function gh {
    $global:LASTEXITCODE = 0
    $calls.Add("gh $($args -join ' ')")
    if ($args[0] -eq 'api') {
        '[{"tag_name":"v0.18.1","published_at":"2026-10-20T00:00:00Z"},{"tag_name":"v0.20.0","published_at":"2026-12-01T00:00:00Z","html_url":"https://github.com/google-ai-edge/LiteRT-LM/releases/tag/v0.20.0","body":"Upstream notes"}]'
    }
}
$oldLocation = $PWD
$savedEnvironment = @{}
foreach ($name in @('GITHUB_REPOSITORY','RUNNER_TEMP','GITHUB_STEP_SUMMARY')) { $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name) }
try {
    Set-Location $sandbox
    $env:GITHUB_REPOSITORY = 'winyunq/LiteRT-LM'
    $env:RUNNER_TEMP = $sandbox
    $env:GITHUB_STEP_SUMMARY = Join-Path $sandbox 'summary.md'
    & $syncScript
    if ([IO.File]::ReadAllText('WRAPPER_VERSION').Trim() -ne '0.20.0') { throw 'Wrong wrapper version' }
    if ([IO.File]::ReadAllText('UPSTREAM_REVISION').Trim() -ne 'upstream-sha') { throw 'Wrong Google source revision' }
    if (!$calls.Contains('git switch --detach refs/google/tags/v0.20.0')) { throw 'Build must start at the Google tag' }
    if (!$calls.Contains('git apply --3way --index ' + (Join-Path $sandbox 'litert-lm-fork.patch'))) { throw 'Fork changes were not applied' }
    if (!$calls.Contains('git push --no-verify origin refs/google/tags/v0.20.0:refs/tags/v0.20.0')) { throw 'Original Google tag was not mirrored' }
    if (!$calls.Contains('gh workflow run build-wrap-windows.yml --repo winyunq/LiteRT-LM --ref codex/version/wrap-v0.20.0')) { throw 'Wrong build source branch' }
    if (@($calls | Where-Object { $_ -match 'push .*(:main|--force)' }).Count) { throw 'Sync must not overwrite main or existing versions' }
    Write-Host 'Release sync contract passed.'
} finally {
    Set-Location $oldLocation
    foreach ($name in $savedEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name]) }
}
