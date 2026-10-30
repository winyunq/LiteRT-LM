# LiteRT-LM fork 与 pack 梳理

对照 Google `2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4`。该提交属于 0.19 开发主线；正式发布仍以 Google release tag 为基线。

## 用户接口顺序

1. FastStart One：最少节点完成单人对话，保持稳定。
2. FastStart Multi：仅增加名字索引，保持稳定。
4. Google 原生 C/C++ API：需要完整能力时直接使用官方接口，版本与官方 SDK 对齐。
3. 扩展 pack：官方能力不能满足用户需求时增加补丁和封装，随上游版本调整，不承诺接口稳定。
5. 全局设置：选择本地模型或在线 API，是独立的设置入口。

## 当前 fork 的实际差异

在 runtime、c、构建配置和上游文档范围内，直接修改了 33 个原有文件，其中 runtime 有 27 个。
此外还有自定义源码、构建脚本、调试产物，以及 Python 原功能的删除。

| 部分 | 当前内容 | 收拢方向 |
| --- | --- | --- |
| 官方原生功能 | `c/engine.h` 已提供 engine create/delete；conversation create/delete/cancel、tools 设置也已存在 | 加载、释放、会话生命周期、工具声明直接使用官方接口，不再各维护一套核心实现 |
| 自研 API 层 | `litert_lm_api.cc/.h` 与 `litert_lm.hpp`，合计约 5,500 行 | 把直接转发官方能力的部分与真正扩展分开，扩展留在 pack |
| 另一套原生桥 | `litert_lm_native_bridge.cc` 约 2,600 行；`litert_lm_native.hpp` 约 730 行 | 与上层自研 API 存在两条封装路线；这轮构建没有使用 native_bridge，保留它并不能减少维护成本 |
| 旧 wrapper | `litert_lm_wrapper.cc/.h`，另有 Gemini 原始副本 | 迁移完成后移除旧路线及备份副本，保留历史版本用于回溯 |
| 严格 GPU 行为 | 修改 embedding lookup、sampler、executor、settings 等多个上游文件 | 单独列为 GPU 核心补丁；与普通模型调用分开，不能仅搬目录便视作纯外挂 |
| KV 导入导出 | `SessionAdvanced`、`SessionInterface`、Conversation 的 session 访问与 skip_prefill | 单独列为 KV 核心补丁，按真实需求维护，不扩散到所有用户入口 |
| Windows 构建兼容 | JSON string_view 转换、onig 静态 CRT、LiteRT 导出/构建设置 | 集中到有版本记录的构建补丁 |
| 无关漂移 | Python 示例、CLI 帮助/环境管理、eval 文件被删除；主 CLI 也被修改 | 恢复 Google 原文件，让用户仍能直接使用原功能 |
| 调试与本机文件 | `WinyunqDebug` 中约 61,000 行日志、exe/obj、Gemini 测试脚本；`.bazelrc` 写死 `D:/bzl_u` | 退出发布源码主线；本机路径留在本机配置或构建参数 |
| 自动同步 | 当前脚本把“整个 fork 对 Google 的 diff”应用到新 tag | 收拢后改为仅携带 pack 和明确列出的补丁，避免继续回放历史杂项 |

MCP 服务切换和路由属于插件的扩展管理能力。工具声明已有官方入口。
模型创建与释放也已有官方入口。它们需要用户接口和生命周期管理，不因此要求复制 Google 的整套实现。

当前 UE 运行层统一通过自研 `LiteRtLm_GetApi` 边界接入。收拢时应把“调用官方现成功能”与“需要修改核心才有的扩展”分开；不是把所有东西换个 pack 目录就完成了。

## 目标源码结构

```text
LiteRT-LM/
  c/、runtime/、python/……      Google 原样源码
  pack/
    native/                   确有需要的扩展
    patches/                  无法通过官方接口完成的最小核心补丁
    build/                    DLL 打包与版本元数据
    README.md                 扩展用途、适用上游版本、接口变动
  .github/workflows/          同步、构建、发布
```

UE 插件继续在自己的仓库中维护 FastStart、蓝图和全局设置；不把插件源码复制进这个 fork。
官方目录跟随 Google。构建时按明确清单应用 pack 补丁；已发布版本保留各自的 Google tag、pack 版本和构建结果。

## 收拢顺序

1. 先确定 pack 边界，停止同时扩张多套 wrapper。
2. 把自研源码、必要补丁和构建脚本集中到 pack。
3. 将官方目录恢复到选定的 Google tag，恢复无关的 Python/CLI 原功能。
4. 现成的加载、释放、会话、工具接口直接接官方；只为真实缺口保留补丁。
5. 删除已替换的旧 wrapper、源码副本和受跟踪的调试产物。
6. 自动同步仅携带 pack，测试通过后产出 DLL/release；插件完成 UE 验证后提交 Fab。

这份文件是当前差异与收拢边界的梳理；尚未执行目录迁移或删除原有实现。
