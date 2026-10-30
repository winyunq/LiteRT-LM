# Wrapper releases

Google release tags are mirrored without changing their source commits. The
hourly `Sync Google release tags` workflow detects new published Google releases
after `v0.18.1`. Existing historical tags are mirrored without rebuilding old SDKs.
Use its manual `upstream_tag` input to prepare a specific historical release.

For each new release, the workflow starts at that exact Google tag and applies
the fork changes relative to the recorded `UPSTREAM_REVISION` on main. It records
the resulting source in `codex/version/wrap-v<version>` and starts the Windows
DLL workflow. A source conflict stops preparation and appears in the workflow log.
Repair the version branch before manually starting its DLL build again.

Only a successful build publishes `wrap-v<version>` and its SDK ZIP, hashes and
release notes. The version branch preserves the customized source; the original
Google tag preserves Google's source. `UPSTREAM_TAG`, `UPSTREAM_REVISION` and
`version.json` identify both versions. Existing version references are retained.

`FAB_UPDATE_NOTES.md` provides English and Chinese version information for the
Unreal plugin release process. A native DLL build verifies the native SDK. The
plugin's Unreal tests, package and Fab review belong to the plugin release.

The sync workflow uses a repository write deploy key in the Actions secret
`UPSTREAM_SYNC_SSH_KEY`, so it can preserve workflow files in Google's tags.
Release uploads and build dispatch use the built-in GitHub token. Builds use
standard hosted runners; no personal access token or local DLL compilation is required.
