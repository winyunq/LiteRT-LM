# LiteRT-LM fork：哪些补丁已经可以退出

对照已同步的 Google 提交 `2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4`（0.19 开发主线）。正式版本仍使用对应的 Google release tag；本次结论不能直接当成较早 tag 的能力清单。

## 结论

目前所需的加载、释放、对话、多会话、克隆、回退、工具声明、分词，官方已经提供接口。这些功能不再构成修改上游核心文件的理由。

先前按“严格 GPU 补丁、KV 补丁”分类保留的方式不合适。应先使用官方能力删除重复实现，剩余确有需求的差异才进入 pack。补丁存在，本身不是保留它的理由。

## 功能对照

| 原先自己补的能力 | 当前官方能力 | 处理结论 |
| --- | --- | --- |
| 模型加载、释放 | `litert_lm_engine_create` / `litert_lm_engine_delete` | 直接使用官方生命周期；UE 负责何时加载、释放，不改引擎内部 |
| 对话创建、释放、流式生成、取消 | `litert_lm_conversation_create` / `delete` / `send_message_stream` / `cancel_process` | FastStart 建在这些接口上；官方取消后会话不可复用，重建逻辑放在插件管理层 |
| 多个角色或命名对话 | 一个 engine 创建多个 conversation；官方维护各自会话状态 | 名字索引属于 UE 管理功能，不需要 KV 核心补丁 |
| 对话分支、复用已有上下文 | `litert_lm_conversation_clone`、`litert_lm_session_clone` | 使用官方 clone，复制预填充状态，不另做一套会话复制 |
| 会话回退 | Session 的 `save_checkpoint`、`rewind_to_checkpoint`、`rewind_to_step` | 使用官方 session 接口；这是 Session 能力，不能冒充 Conversation 的 JSON 历史回退接口 |
| 分词、反分词 | `litert_lm_engine_tokenize` / `litert_lm_engine_detokenize` | 删除重复实现，只保留需要的 UE 参数转换 |
| 工具声明和模型输出工具调用 | `litert_lm_conversation_config_set_tools` 与官方消息格式 | 原生工具能力直接复用；MCP 连接、服务选择和调用路由留在插件扩展层 |
| 追加输入而暂不生成 | C++ `OptionalArgs.has_pending_message`：执行 prefill、跳过 decode | 已有官方路径；不需要为此修改 Conversation。它不等于旧补丁的“连 prefill 也跳过” |
| 兼容旧 GPU sampler 缺少符号 | 当前 `top_k_webgpu_sampler.h/.cc` 已导出、实现 `UpdateConfig`、`CanHandleInput`、`HandlesInput`、`SetInferenceFuncAndInputTensors` | 同版本源码构建不需要这套旧 DLL 兼容逻辑；恢复官方加载行为 |

MCP 服务热切换不等于官方已经提供任意修改运行中会话工具表的接口。当前 C 接口的工具入口作用于 conversation config；服务连接与路由由 pack/插件管理，需要更换工具声明时使用官方配置和会话生命周期。

## 两个不能混同的额外行为

### KV 字节导入导出

官方 clone 和 checkpoint 解决内存中的复制、分支和回退，不能直接等同于把完整状态导出为字节、跨进程保存并恢复。

我们的 `GetKVCache` / `SetKVCache`、`GetSession`、`skip_prefill` 以及暴露内部 processed context 的修改，是旧 wrapper 为这条额外路径添加的。旧 `ClearCachedEnvironments` 现在也只是空操作；引擎拥有自己的环境生命周期。

当前 `litert_lm_api.cc` 新接口没有调用这些 KV 钩子；调用仍在 `litert_lm_wrapper.cc` 的旧入口中。但是 `runtime/engine/BUILD` 的 `litert_lm_wrapper` 目标同时编译新、旧两个实现，所以旧入口仍把补丁带进当前 DLL。

因此，普通单人和命名多人对话不应继续依赖这些修改。收拢时移除旧 KV 入口及其核心钩子；字节状态保存若成为独立需求，再作为第 3 类实验扩展处理。

### 强制完全 GPU 执行

选择 GPU 后端，与“所有相关子模型都必须证明完全加速，否则报错”是不同语义。

后者是我们新增的 `require_full_acceleration`，涉及 embedding、executor、settings 和 sampler 的 CPU 回退行为。底层 LiteRT 已有 `CompiledModel::IsFullyAccelerated()` 查询，但这不等于 LiteRT-LM 官方已经提供同样的强制策略。

新 wrapper 当前确实主动启用了该策略。它应按独立产品需求决定是否保留，不能作为 FastStart 或普通 GPU 调用的默认前提，也不能混进“官方缺少加载和会话管理”的理由。

## 与功能无关的漂移

此前统计的 runtime、c、构建配置和上游文档范围中，有 33 个原文件被修改，其中 runtime 有 27 个。不能把这些改动整体视作仍然必要的功能包。

| 内容 | 收拢方向 |
| --- | --- |
| 自研 API、另一套 native bridge、旧 wrapper、Gemini 源码副本 | 对外需要的 UE 接口继续由插件维护；内部接入收敛到官方 API，删除被替代路线和源码副本 |
| Python 示例、CLI、eval 等原功能删除或改写 | 恢复官方原功能，与扩展解耦 |
| 调试日志、exe/obj、本机 `D:/bzl_u` 路径 | 从发布源码移除，本机配置留在本机 |
| JSON 的 MSVC 字符串转换、静态 CRT、链接与 DLL 导出设置 | 属于 Windows 构建适配，不是模型功能缺口；单独记录最小构建差异 |
| 自动同步回放整个 fork diff | 收拢后仅携带 pack 和明确列出的必要构建补丁，避免再次回放淘汰的旧实现 |

## pack 的边界

Google 原目录跟随上游。pack 只承载实际扩展、必要的 Windows 构建适配和发布脚本；UE 仓库维护蓝图、全局设置、多语言文档与 MCP 路由。

用户接口按 `1 → 2 → 4 → 3` 使用：最简单对话、命名多会话、官方原生 API、实验扩展。第 5 类提供全局后端设置。第 1、2 类的稳定来自插件对用户保持接口契约，不来自冻结一份自改的 Google 核心。

## 已执行的收拢

- Google 原有的 runtime、c、Python、CLI、README 和相关构建文档已恢复到上述上游提交。
- 当前接入层移至 `pack/native`；旧 wrapper、KV 钩子、重复 native bridge 和 Gemini 源码副本已移除。
- 严格 GPU 策略和旧 sampler 兼容补丁已移除，GPU 采用官方行为。
- Windows 构建差异存入 `pack/patches`，由云端构建应用；官方目录不再持久携带这些差异。
- 发布同步仅携带 pack 和明确列出的自动化、版本文件，不再回放整个 fork 的差异。
- 新开发包版本为 `0.19.0-dev.2`。Windows 补丁应用检查与 release 同步脚本测试已通过；DLL 构建结果以该版本对应的 GitHub Actions 为准。

上文旧入口和旧补丁的描述是收拢前的审计依据。调试产物的清理不属于这次核心接入变更。

## 对照入口

- [官方 C engine 接口](https://github.com/google-ai-edge/LiteRT-LM/blob/2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4/c/engine.h)：生命周期、session clone、checkpoint、分词。
- [官方 C conversation 接口](https://github.com/google-ai-edge/LiteRT-LM/blob/2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4/c/conversation.h)：会话、工具配置、clone、流式消息。
- [官方 C++ conversation 接口](https://github.com/google-ai-edge/LiteRT-LM/blob/2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4/runtime/conversation/conversation.h)：history、has_pending_message、clone。
- [官方 WebGPU sampler 接口](https://github.com/google-ai-edge/LiteRT-LM/blob/2c1c9a67cbd3a12e20fdf1d750855e6afd5649c4/runtime/components/top_k_webgpu_sampler.h)：当前 sampler 导出。
