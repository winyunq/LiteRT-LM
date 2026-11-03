param([string]$UpstreamTag)
$ErrorActionPreference = 'Stop'
$repo = $env:GITHUB_REPOSITORY
$upstream = 'google-ai-edge/LiteRT-LM'

function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args[0]) failed ($LASTEXITCODE)" }
}
function Api([string]$Path) {
    $json = & gh api $Path
    if ($LASTEXITCODE -ne 0) { throw "GitHub API failed: $Path" }
    return ($json | ConvertFrom-Json)
}

$releases = @(Api "repos/$upstream/releases?per_page=100")
$currentVersion = [semver]([IO.File]::ReadAllText('WRAPPER_VERSION').Trim())
if ($UpstreamTag) {
    $releases = @($releases | Where-Object tag_name -eq $UpstreamTag)
    if (!$releases.Count) { throw "Google release not found: $UpstreamTag" }
}
$pending = @($releases |
    Where-Object { !$_.draft -and !$_.prerelease -and $_.immutable -eq $true } |
    Where-Object { [semver]($_.tag_name -replace '^v', '') -gt $currentVersion } |
    Sort-Object { [semver]($_.tag_name -replace '^v', '') })

if (!$UpstreamTag -and $pending.Count) {
    $prepared = @(Api "repos/$repo/git/matching-refs/heads/codex/version/wrap-v")
    $pending = @($pending | Where-Object {
        $branch = 'refs/heads/codex/version/wrap-v' + ($_.tag_name -replace '^v', '')
        $branch -notin $prepared.ref
    })
}
if (!$pending.Count) {
    'No new immutable Google release to synchronize.' >> $env:GITHUB_STEP_SUMMARY
    return
}

Invoke-Git config user.name 'github-actions[bot]'
Invoke-Git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
Invoke-Git config credential.helper '!gh auth git-credential'
Invoke-Git remote add google "https://github.com/$upstream.git"
foreach ($release in $pending) {
    $tag = $release.tag_name
    Invoke-Git fetch google --filter=blob:none "+refs/tags/${tag}:refs/google/tags/$tag"
    Invoke-Git push --no-verify origin "refs/google/tags/${tag}:refs/tags/$tag"
}

$forkHead = (Invoke-Git rev-parse HEAD).Trim()
$base = [IO.File]::ReadAllText('UPSTREAM_REVISION').Trim()
$patch = Join-Path $env:RUNNER_TEMP 'litert-lm-pack.patch'
Invoke-Git diff --binary "--output=$patch" $base $forkHead -- pack .github/scripts .github/workflows/build-wrap-windows.yml .github/workflows/build-wrap-platforms.yml .github/workflows/sync-upstream-releases.yml .github/upstream-release-baseline.txt .lfsconfig UPSTREAM_REVISION WRAPPER_VERSION WRAPPER_RELEASES.md

foreach ($release in $pending) {
    $tag = $release.tag_name
    $version = $tag -replace '^v', ''
    $branch = "codex/version/wrap-v$version"
    $existing = @(Invoke-Git ls-remote --heads origin "refs/heads/$branch")
    # A prepared version is built once. Failed builds remain available for repair.
    if ($existing.Count -and !$UpstreamTag) { continue }
    if (!$existing.Count) {
        $tagRef = "refs/google/tags/$tag"
        $upstreamRevision = (Invoke-Git rev-parse "${tagRef}^{commit}").Trim()
        Invoke-Git switch --detach $tagRef
        Invoke-Git apply --3way --index $patch
        [IO.File]::WriteAllText('WRAPPER_VERSION', "$version`n")
        [IO.File]::WriteAllText('UPSTREAM_REVISION', "$upstreamRevision`n")
        [IO.File]::WriteAllText('UPSTREAM_TAG', "$tag`n")
        $notes = "# Google LiteRT-LM $tag`n`nSource: $($release.html_url)`n`n$($release.body)`n"
        [IO.File]::WriteAllText('UPSTREAM_RELEASE_NOTES.md', $notes)
        Invoke-Git add -f WRAPPER_VERSION UPSTREAM_REVISION UPSTREAM_TAG UPSTREAM_RELEASE_NOTES.md
        Invoke-Git commit -m "Prepare wrapper from Google $tag with extension pack"
        Invoke-Git switch -c $branch

        # Import only the Windows files used by the DLL build, from Google to the fork.
        Invoke-Git -c "lfs.url=https://github.com/$upstream.git/info/lfs" lfs fetch google $tagRef '--include=prebuilt/windows_x86_64/**'
        $oids = @(Invoke-Git lfs ls-files -l '--include=prebuilt/windows_x86_64/**' | ForEach-Object { ($_ -split ' ')[0] } | Sort-Object -Unique)
        if ($oids.Count) { Invoke-Git lfs push --object-id origin @oids }
        Invoke-Git push --no-verify origin "HEAD:refs/heads/$branch"
    }
    & gh workflow run build-wrap-platforms.yml --repo $repo --ref $branch
    if ($LASTEXITCODE -ne 0) { throw "Could not start native SDK builds for $tag" }
    "Prepared $branch from $tag" >> $env:GITHUB_STEP_SUMMARY
}
