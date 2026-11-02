$ErrorActionPreference = 'Stop'
$version = [IO.File]::ReadAllText('WRAPPER_VERSION').Trim()
$repo = $env:GITHUB_REPOSITORY
$platforms = if ($env:WRAPPER_PLATFORMS) { $env:WRAPPER_PLATFORMS } else { 'Win64' }
$sourceSha = (git rev-parse HEAD).Trim()
$tag = "wrap-v$version"
if ($env:GITHUB_REF_TYPE -eq 'tag' -and $env:GITHUB_REF_NAME -ne $tag) { throw 'Tag and WRAPPER_VERSION differ' }
$branch = "codex/version/$tag"
$heads = gh api "repos/$repo/git/matching-refs/heads/$branch" | ConvertFrom-Json
if ($LASTEXITCODE -ne 0) { throw 'Could not read version branches' }
$head = @($heads | Where-Object ref -eq "refs/heads/$branch")
if ($head.Count) {
    if ($head[0].object.sha -ne $sourceSha) { throw "Version branch already points to another commit: $branch" }
} else {
    gh api "repos/$repo/git/refs" -X POST -f "ref=refs/heads/$branch" -f "sha=$sourceSha"
    if ($LASTEXITCODE -ne 0) { throw 'Could not archive version branch' }
}

$upstreamRevision = [IO.File]::ReadAllText('UPSTREAM_REVISION').Trim()
$upstreamTag = if (Test-Path -LiteralPath UPSTREAM_TAG) { [IO.File]::ReadAllText('UPSTREAM_TAG').Trim() } else { 'development main' }
$notes = @"
# LiteRT-LM wrapper $version

- Google source: $upstreamTag ($upstreamRevision).
- Fork source: $sourceSha.
- Native SDK platforms built from source: $platforms. Core and wrapper exports checked in each platform job.
- Each SDK ZIP includes public headers, native runtime libraries and version.json with SHA-256 hashes; Windows also includes import libraries.

This is a native SDK release. Unreal plugin testing and Fab submission are separate steps.
"@
if (Test-Path -LiteralPath UPSTREAM_RELEASE_NOTES.md) {
    $notes += "`n`n" + [IO.File]::ReadAllText('UPSTREAM_RELEASE_NOTES.md')
}
[IO.File]::WriteAllText('out/RELEASE_NOTES.md', $notes)
$fab = @"
# Fab 更新说明素材 / Fab update notes

LiteRT-LM wrapper: $version
Google source: $upstreamTag
Fork source: $sourceSha

底层运行库已更新至上述 Google 源码版本，并保留本插件的封装修改。
The native runtime follows the Google source version above and retains our wrapper changes.

本次 GitHub 验证：$platforms 原生库编译、导出符号检查及 SDK 打包。
Verified here: $platforms native library compilation, export checks and SDK packaging.

这是供插件发布流程使用的更新说明素材，不表示 Unreal 测试或 Fab 审核已经通过。
These notes do not record Unreal test results or Fab approval.
"@
[IO.File]::WriteAllText('out/FAB_UPDATE_NOTES.md', $fab)
$assets = @((Get-ChildItem -LiteralPath out -Filter *.zip).FullName) + @('out/FAB_UPDATE_NOTES.md', 'out/RELEASE_NOTES.md')
gh release view $tag --repo $repo *> $null
if ($LASTEXITCODE -eq 0) {
    gh release upload $tag @assets --repo $repo --clobber
} else {
    $releaseArgs = @('release', 'create', $tag) + $assets + @('--repo', $repo, '--target', $sourceSha, '--title', "LiteRT-LM wrapper $version", '--notes-file', 'out/RELEASE_NOTES.md')
    if ($version.Contains('-')) { $releaseArgs += '--prerelease' }
    gh @releaseArgs
}
if ($LASTEXITCODE -ne 0) { throw 'Could not publish wrapper release' }
