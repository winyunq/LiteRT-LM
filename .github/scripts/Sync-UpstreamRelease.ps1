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

Invoke-Git config user.name 'github-actions[bot]'
Invoke-Git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
Invoke-Git config credential.helper '!gh auth git-credential'
Invoke-Git remote add google "https://github.com/$upstream.git"
Invoke-Git fetch google --filter=blob:none '+refs/tags/*:refs/google/tags/*'
$forkTags = @{}
foreach ($line in @(Invoke-Git ls-remote --tags origin)) {
    $fields = $line -split '\s+'
    $forkTags[$fields[1]] = $fields[0]
}
foreach ($line in @(Invoke-Git for-each-ref '--format=%(objectname) %(refname)' refs/google/tags/)) {
    $oid, $sourceRef = $line -split ' ', 2
    $tag = $sourceRef.Substring('refs/google/tags/'.Length)
    $destination = "refs/tags/$tag"
    if ($forkTags.ContainsKey($destination)) {
        if ($forkTags[$destination] -ne $oid) { throw "Tag differs from Google: $tag" }
        continue
    }
    # Original tags retain Google's LFS endpoint; there are no fork LFS uploads.
    Invoke-Git push --no-verify origin "${sourceRef}:$destination"
}

$releases = @(Api "repos/$upstream/releases?per_page=100")
$baseline = [IO.File]::ReadAllText('.github/upstream-release-baseline.txt').Trim()
$baselineRelease = $releases | Where-Object tag_name -eq $baseline | Select-Object -First 1
if (!$baselineRelease) { throw "Release baseline not found: $baseline" }
if ($UpstreamTag) {
    $pending = @($releases | Where-Object tag_name -eq $UpstreamTag)
    if (!$pending.Count) { throw "Google release not found: $UpstreamTag" }
} else {
    $pending = @($releases | Where-Object { !$_.draft -and $_.published_at -gt $baselineRelease.published_at } | Sort-Object published_at)
}

$forkHead = (Invoke-Git rev-parse HEAD).Trim()
$base = [IO.File]::ReadAllText('UPSTREAM_REVISION').Trim()
$patch = Join-Path $env:RUNNER_TEMP 'litert-lm-fork.patch'
Invoke-Git diff --binary "--output=$patch" $base $forkHead

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
        $buildInfo = "#pragma once`n#define LITERT_LM_WRAPPER_VERSION `"$version`"`n#define LITERT_LM_UPSTREAM_REVISION `"$upstreamRevision`"`n"
        [IO.File]::WriteAllText('runtime/engine/litert_lm_build_info.h', $buildInfo)
        $notes = "# Google LiteRT-LM $tag`n`nSource: $($release.html_url)`n`n$($release.body)`n"
        [IO.File]::WriteAllText('UPSTREAM_RELEASE_NOTES.md', $notes)
        Invoke-Git add -f WRAPPER_VERSION UPSTREAM_REVISION UPSTREAM_TAG UPSTREAM_RELEASE_NOTES.md runtime/engine/litert_lm_build_info.h
        Invoke-Git commit -m "Prepare wrapper from Google $tag with fork changes"
        Invoke-Git switch -c $branch

        # Import only the Windows files used by the DLL build, from Google to the fork.
        Invoke-Git -c "lfs.url=https://github.com/$upstream.git/info/lfs" lfs fetch google $tagRef '--include=prebuilt/windows_x86_64/**'
        $oids = @(Invoke-Git lfs ls-files -l '--include=prebuilt/windows_x86_64/**' | ForEach-Object { ($_ -split ' ')[0] } | Sort-Object -Unique)
        if ($oids.Count) { Invoke-Git lfs push --object-id origin @oids }
        Invoke-Git push --no-verify origin "HEAD:refs/heads/$branch"
    }
    & gh workflow run build-wrap-windows.yml --repo $repo --ref $branch
    if ($LASTEXITCODE -ne 0) { throw "Could not start DLL build for $tag" }
    "Prepared $branch from $tag" >> $env:GITHUB_STEP_SUMMARY
}
