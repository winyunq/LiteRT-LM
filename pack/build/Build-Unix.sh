#!/usr/bin/env bash
set -euo pipefail

platform="$1"
config="$2"
out="out/LiteRtLm"
bin="$out/Binaries/$platform"
mkdir -p "$bin" "$out/Include/c"
options=(-c opt "--config=$config" --deleted_packages=swift
  "--disk_cache=$HOME/.cache/litert-lm/disk"
  "--repository_cache=$HOME/.cache/litert-lm/repository")
if [[ "$platform" == Android-* ]]; then
  options+=(--define=litert_runtime_link_mode=dynamic)
fi

bazel build "${options[@]}" //c:litert-lm //pack/native:litert_lm_wrapper
exec_root="$(bazel info execution_root)"
core="$(bazel cquery //c:litert-lm "${options[@]}" --output=files)"
wrapper="$(bazel cquery //pack/native:litert_lm_wrapper "${options[@]}" --output=files)"
cp -L "$exec_root/$core" "$exec_root/$wrapper" "$bin/"

bazel cquery 'deps(set(//c:litert-lm //pack/native:litert_lm_wrapper))' "${options[@]}" --notool_deps --output=files > "$RUNNER_TEMP/runtime-files.txt"
while IFS= read -r file; do
  case "$file" in
    runtime/components/stub/*) continue ;;
    *.so|*.so.*|*.dylib)
      if [[ -f "$exec_root/$file" ]]; then cp -L "$exec_root/$file" "$bin/"; fi ;;
  esac
done < "$RUNNER_TEMP/runtime-files.txt"

case "$platform" in
  Mac-*|IOS-*)
    nm -gU "$bin/$(basename "$core")" > "$RUNNER_TEMP/core-exports.txt"
    nm -gU "$bin/$(basename "$wrapper")" > "$RUNNER_TEMP/wrapper-exports.txt"
    ;;
  Android-*)
    nm_tool="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm"
    "$nm_tool" -D --defined-only "$bin/$(basename "$core")" > "$RUNNER_TEMP/core-exports.txt"
    "$nm_tool" -D --defined-only "$bin/$(basename "$wrapper")" > "$RUNNER_TEMP/wrapper-exports.txt"
    ;;
  *)
    nm -D --defined-only "$bin/$(basename "$core")" > "$RUNNER_TEMP/core-exports.txt"
    nm -D --defined-only "$bin/$(basename "$wrapper")" > "$RUNNER_TEMP/wrapper-exports.txt"
    ;;
esac
grep -Eq '[[:space:]]_?litert_lm_get_api_version([[:space:]]|$)' "$RUNNER_TEMP/core-exports.txt"
grep -Eq '[[:space:]]_?LiteRtLm_GetApi([[:space:]]|$)' "$RUNNER_TEMP/wrapper-exports.txt"

for header in api_export.h conversation.h embedding_engine.h engine.h error_reporter.h experimental.h model_info.h; do
  cp "c/$header" "$out/Include/c/"
done
cp pack/native/litert_lm_api.h pack/native/litert_lm.hpp "$out/Include/"
cp LICENSE "$out/"
version="$(tr -d '\r\n' < WRAPPER_VERSION)"
upstream_version="$(sed -n 's/^VERSION = "\([^"]*\)".*/\1/p' version.bzl)"
c_api_version="$(sed -n 's/^C_API_VERSION = "\([^"]*\)".*/\1/p' version.bzl)"
(
  cd "$out"
  find . -type f | LC_ALL=C sort | while IFS= read -r file; do
    digest="$(shasum -a 256 "$file" | awk '{print $1}')"
    jq -nc --arg path "${file#./}" --arg sha256 "$digest" '{path:$path,sha256:$sha256}'
  done
) | jq -s . > "$RUNNER_TEMP/sdk-hashes.json"
jq -n --arg wrapper_version "$version" --arg upstream_version "$upstream_version" \
  --arg upstream_commit "$(tr -d '\r\n' < UPSTREAM_REVISION)" --arg source_commit "$GITHUB_SHA" \
  --arg c_api_version "$c_api_version" --arg build "$platform Release" \
  --slurpfile files "$RUNNER_TEMP/sdk-hashes.json" \
  '{wrapper_version:$wrapper_version,upstream_version:$upstream_version,
    upstream_commit:$upstream_commit,source_commit:$source_commit,
    c_api_version:$c_api_version,build:$build,files:$files[0]}' > "$out/version.json"
(cd out && zip -qr "LiteRtLm-$version-$platform.zip" LiteRtLm)
