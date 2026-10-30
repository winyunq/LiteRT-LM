# LiteRT-LM extension pack

Google's `c/`, `runtime/`, `python/` and original build configuration follow `UPSTREAM_REVISION` unchanged.

- `native/`: the current `LiteRtLm_GetApi` interface used by the UE plugin, implemented with official LiteRT-LM C++ APIs. Native users can also call the separately packaged official C DLL directly.
- `patches/`: Windows build compatibility only: two MSVC JSON key conversions, static CRT and LiteRT link/export settings. `windows-build.patch` is applied by the Windows build script to its checkout; it contains no KV or GPU execution policy changes.
- `build/Build-Windows.ps1`: builds the official C DLL and extension DLL, checks their exports, and packages the SDK and version metadata.

The old wrapper, raw KV import/export, duplicate native bridge and strict full-GPU policy have been removed. GPU backend selection follows official behavior.

GitHub release sync starts from the exact Google release tag and adds this pack and the listed automation files. It does not replay arbitrary changes under Google's source directories. Published wrapper releases retain their own version branch and source revision.

Local DLL compilation is not required; GitHub Actions runs `build-wrap-windows.yml`. UE FastStart interfaces, backend settings, MCP routing and multilingual documentation remain in the UE plugin repository.
