// Copyright (c) 2025-2026 Winyunq. All rights reserved.
//
// DEPRECATED LEGACY SOURCE API. Its single-global-conversation ownership model
// is intentionally not the canonical SDK and it is not ABI-compatible with
// LiteRtLm_Extension.lib. New C/C++ code must use litert_lm_native_api.h (the
// complete official API) or the versioned litert_lm_api.h extension table.
#ifndef LITERT_LM_WRAPPER_H
#define LITERT_LM_WRAPPER_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
  #ifdef LITERT_LM_WRAPPER_IMPLEMENTATION
    // Windows exports are controlled exclusively by litert_lm_wrapper.def.
    #define DLL_EXPORT
  #else
    #define DLL_EXPORT __declspec(dllimport)
  #endif
#else
  #define DLL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

    // --- 数据结构 ---

    /**
     * @brief       LiteRT-LM 引擎的全局初始化配置参数结构体
     *              用于在创建引擎实例时配置模型路径、硬件后端以及多模态等核心参数。
     **/
    typedef struct {
        /// 指向模型文件路径的只读字符串指针，支持的格式为 *.litert_lm 的绝对路径或相对路径。
        const char* model_path;

        /// 指定模型推理所使用的硬件加速后端名称，例如 "cpu" 代表使用处理器，"gpu" 代表使用图形处理器加速推理。
        const char* backend;

        /// 以 JSON 格式表达的可用工具函数声明列表字符串，模型在需要进行外部工具调用 (Tool Calling / Function Calling) 时解析，传入 nullptr 则代表禁用工具。
        const char* tools_json;

        /// 单次推理生成的最大 Token 数量限制，用于防止模型在推理生成时出现无限循环输出。
        int max_num_tokens;

        /// 推理时允许使用的 CPU 线程数量，主要用于 CPU 后端的并行计算优化，设置合适数值可充分利用多核处理器性能。
        int num_threads;

        /// 预填充 (Prefill) 阶段的 Token 分块处理大小 (Chunk Size)，通过控制预填充的单次处理量来平衡显存/内存的峰值占用与预填充速度。
        int prefill_chunk_size;

        /// 是否开启性能基准测试统计标志，占 1 位。值为 1 表示开启，引擎将统计并打印 Prefill 与 Decode 阶段的详细耗时信息。
        unsigned int bEnableBenchmark         : 1;

        /// 是否开启图形着色器优化标志，占 1 位。在 GPU 加速后端上启用时，会对运行的 Shader 进行编译期优化以提升运行效率。
        unsigned int bOptimizeShader          : 1;

        /// 是否启用视觉多模态输入支持标志，占 1 位。值为 1 时，引擎可以接收并处理包含图像数据的多模态消息。
        unsigned int bEnableVision            : 1;

        /// 是否启用音频多模态输入支持标志，占 1 位。值为 1 时，引擎可以接收并处理包含音频数据的多模态消息。
        unsigned int bEnableAudio             : 1;

        /// 是否在多个引擎实例间共享常量张量内存标志，占 1 位。在多实例或并发运行场景下，启用可显著降低显存和系统内存的重复占用。
        unsigned int bShareConstantTensors    : 1;

        /// 是否开启主机映射内存指针 (Host Mapped Pointer) 标志，占 1 位。启用后允许直接将 CPU 内存地址映射给 GPU 使用，减少数据在 CPU 和 GPU 之间的拷贝开销。
        unsigned int bEnableHostMappedPointer : 1;
    } LiteRtLm_Config;

    /**
     * @brief       控制文本生成随机性与约束规则的采样参数结构体
     *              包含了常见的 Temperature、Top-P、Top-K 以及结构化解码 (Constrained Decoding) 约束。
     **/
    typedef struct {
        /// 采样温度值 (Temperature)，值越高生成的随机性、多样性越强；值越低则生成的确定性越强。通常范围在 0.0 到 2.0 之间。
        float temperature;

        /// Top-P 概率累计阈值 (核采样 Nucleus Sampling)，在概率分布中，仅保留累加概率达到该阈值的最可能 Token 候选集，若设为 1.0 则代表不使用 Top-P 过滤。
        float top_p;

        /// Top-K 候选 Token 数量限制，在预测下一个 Token 时，仅在概率最高的 K 个候选 Token 中进行采样，若为 0 或负数则表示不限制。
        int top_k;

        /// 本次推理生成任务允许输出的最大 Token 数量，控制该条响应内容的最大长度限制。
        int max_tokens;

        /// 强制结构化约束类型 (Constrained Decoding)。0: 无任何格式约束，1: 通过正则表达式进行生成约束，2: 通过 JSON Schema 架构约束，3: 通过 Lark 语法规则约束。
        int constraint_type;

        /// 约束条件具体定义的只读字符串内容指针，根据 constraint_type 的不同，传入正则表达式、JSON Schema 的 JSON 串或 Lark 规约字符串；不约束时为 nullptr。
        const char* constraint_string;
    } LiteRtLm_SamplingParams;

    /**
     * @brief       单次增量推理生成返回的结果信息结构体
     *              包含生成的新文本片段、原始完整 JSON 数据、错误信息及状态标识。
     **/
    typedef struct {
        /// 指向当前增量生成的纯文本片段，只读字符串指针。警告：此指针指向的内存仅在回调函数执行期间保证有效，离开回调后切勿引用！
        const char* text_chunk;

        /// 指向包含完整原始生成数据的 JSON 字符串指针，包含了模型输出的所有底层字段 (如 tool_calls、channels、原始 content 等)，便于深度解析。
        const char* full_json_chunk;

        /// 指向错误描述信息的只读字符串指针，当推理过程中发生异常或错误时指向具体错误说明，正常生成时为 nullptr。
        const char* error_msg;

        /// 推理生成任务是否已完全结束的布尔标识，值为 1 表示生成任务已全部执行完毕，后续将不再触发回调；值为 0 表示正在生成中。
        int bIsDone;

        /// 当前推理生成的实时生成速度，表示当前生成过程中的实际吞吐率，单位为：每秒生成的 Token 数量 (Tokens Per Second)。
        float tokens_per_sec;
    } LiteRtLm_Result;

    /**
     * @brief       增量生成结果的异步回调函数指针类型定义
     *
     * @param       参数名称: result                        数据类型:        LiteRtLm_Result
     *              参数说明: 包含当前增量生成的文本片段、JSON原始信息、错误提示及结束标志的结构体。
     *
     * @param       参数名称: user_ptr                      数据类型:        void*
     *              参数说明: 外部调用方在触发推理时传入的自定义上下文指针，通常用于在回调中还原上下文对象。
     **/
    typedef void (*LiteRtLmCallback)(LiteRtLm_Result result, void* user_ptr);

    // --- 0. 硬件与探测接口 ---

    /**
     * @brief       获取当前系统环境下所有可用的推理加速后端列表
     *
     * @return      返回指向以逗号分隔的可用后端名称列表的只读静态字符串指针 (例如 "cpu,gpu")，无需外部释放。
     *              数据类型: const char*
     **/
    DLL_EXPORT const char* LiteRtLm_GetAvailableBackends();

    /** ABI contract version. Version 2 adds explicit Conversation reset and
     * runtime evidence queries. */
    DLL_EXPORT int LiteRtLm_GetApiVersion();

    /** Human-readable immutable build/capability string. */
    DLL_EXPORT const char* LiteRtLm_GetBuildInfo();

    // --- 1. 引擎生命周期与单会话状态接口 ---

    /**
     * @brief       初始化创建底层的 LiteRT-LM 引擎实例
     *              该函数会在引擎内部自动隐式创建唯一的全局对话状态上下文与活跃的 KV 缓存 (Key-Value Cache) 空间。
     *
     * @param       参数名称: config                        数据类型:        LiteRtLm_Config
     *              参数说明: 引擎初始化配置参数，包含模型路径、加速后端、线程数、块大小等配置。
     *
     * @return      成功时返回指向引擎实例的通用 void* 指针；失败时返回 nullptr。
     *              数据类型: void*
     **/
    DLL_EXPORT void* LiteRtLm_CreateEngine(LiteRtLm_Config config);

    /**
     * @brief       销毁指定的 LiteRT-LM 引擎实例
     *              释放引擎占用的显存、系统内存等全部资源，同时自动隐式销毁内部唯一的对话状态及 KV 缓存。
     *
     * @param       参数名称: engine_ptr                    数据类型:        void*
     *              参数说明: 指向待销毁引擎实例的通用指针 (即 LiteRtLm_CreateEngine 的返回值)。
     **/
    DLL_EXPORT void LiteRtLm_DestroyEngine(void* engine_ptr);

    /**
     * @brief Recreate the Conversation/Session while preserving the Engine,
     * model weights, resolved backend and the tools preface supplied at engine
     * creation. This is the only reset API; SetKVCache(nullptr, 0) is not a
     * reset and retains logical history.
     *
     * Must be called while no inference is active. Returns 0 on success.
     */
    DLL_EXPORT int LiteRtLm_ResetConversation(void* engine_ptr);

    /** Number of logical messages in the active Conversation, or -1 on error. */
    DLL_EXPORT long long LiteRtLm_GetConversationHistorySize(void* engine_ptr);

    /** Monotonic Conversation generation for the current Engine (starts at 1). */
    DLL_EXPORT uint64_t LiteRtLm_GetConversationGeneration(void* engine_ptr);

    /**
     * Backend resolved into the live Engine settings (for example "gpu").
     * This is stronger than a caller config echo, but is not low-level GPU
     * driver utilization telemetry.
     */
    DLL_EXPORT const char* LiteRtLm_GetResolvedBackend(void* engine_ptr);

    /** Number of tool schemas parsed into the persistent conversation preface. */
    DLL_EXPORT int LiteRtLm_GetLoadedToolsCount(void* engine_ptr);

    /**
     * @brief       向全局会话中追加一条最新的用户消息
     *              支持传入复杂的 JSON 格式以实现文本、图像等多模态数据的混合输入。
     *
     * @param       参数名称: json_msg                      数据类型:        const char*
     *              参数说明: 符合多模态协议的 JSON 字符串，例如：{"role": "user", "content": [{"type": "text", "text": "你好"}]}。
     **/
    DLL_EXPORT void LiteRtLm_AppendUserMessage(const char* json_msg);

    /**
     * @brief       向全局会话中追加一条历史对话消息，并立即触发预填充建立 KV 缓存
     *              此函数用于同步已有的对话历史。它会立即触发引擎的预填充(Prefill)逻辑以建立 KV Cache，但不会触发生成过程。
     *
     *              Demo (恢复历史并提问):
     *              @code
     *              // 1. 恢复历史 (立即建立 KV Cache)
     *              LiteRtLm_AppendHistoryMessage("{\"role\": \"system\", \"content\": \"你是一个助手\"}");
     *              LiteRtLm_AppendHistoryMessage("{\"role\": \"user\", \"content\": \"你好\"}");
     *              LiteRtLm_AppendHistoryMessage("{\"role\": \"assistant\", \"content\": \"你好！有什么可以帮你的？\"}");
     *
     *              // 2. 准备新问题 (覆盖式准备，不立即执行)
     *              LiteRtLm_AppendUserMessage("{\"role\": \"user\", \"content\": \"今天天气怎么样？\"}");
     *
     *              // 3. 触发推理 (基于历史缓存秒回)
     *              LiteRtLm_RunInference(params, callback, user_ptr);
     *              @endcode
     *
     * @param       参数名称: json_msg                      数据类型:        const char*
     *              参数说明: 包含 role 的完整 JSON 消息字符串。
     **/
    DLL_EXPORT void LiteRtLm_AppendHistoryMessage(const char* json_msg);

    /**
     * Legacy ABI compatibility helper. Converts plain text to an assistant
     * history message and performs the same synchronous pending prefill as
     * LiteRtLm_AppendHistoryMessage.
     */
    DLL_EXPORT void LiteRtLm_AppendAssistantMessage(const char* text);

    /**
     * @brief       基于当前全局会话已有的上下文历史，异步触发增量推理生成响应
     *              本调用为非阻塞式的，推理任务会提交给后台，生成的结果会通过传入的回调函数分发。
     *
     * @param       参数名称: params                        数据类型:        LiteRtLm_SamplingParams
     *              参数说明: 采样和结构化约束参数，控制温度、Top-K、Top-P 以及强制输出的格式。
     *
     * @param       参数名称: callback                      数据类型:        LiteRtLmCallback
     *              参数说明: 接收推理增量结果的回调函数指针。
     *
     * @param       参数名称: user_ptr                      数据类型:        void*
     *              参数说明: 用户自定义的上下文指针，在回调函数执行时会原样传回，方便调用方管理业务关联状态。
     **/
    DLL_EXPORT void LiteRtLm_RunInference(
        LiteRtLm_SamplingParams params,
        LiteRtLmCallback callback,
        void* user_ptr
    );

    /**
     * @brief       立即中断当前正在异步执行的全局增量推理生成任务
     *              调用后，生成进程将被强行中止，且不会再触发后续的回调函数。
     **/
    DLL_EXPORT void LiteRtLm_StopMessage();

    /**
     * @brief       阻塞当前线程，等待引擎完成所有后台的异步推理与生成任务。
     *              RunInference 是非阻塞的（仅提交任务到 GPU/CPU 队列），必须在 RunInference 之后调用此函数驱动回调分发。
     *
     * @param       参数名称: engine_ptr                    数据类型:        void*
     *              参数说明: CreateEngine 返回的引擎指针。
     *
     * @param       参数名称: timeout_sec                   数据类型:        int
     *              参数说明: 超时秒数 (<=0 使用引擎默认超时 10 分钟)。
     *
     * @return      返回等待状态码：0 代表在超时前成功结束；1 代表等待超时；-1 代表发生内部错误。
     *              数据类型: int
     **/
    DLL_EXPORT int LiteRtLm_WaitUntilDone(void* engine_ptr, int timeout_sec);

    // --- 2. GPU 物理 KV 缓存直接锁显存导出与恢复接口 ---

    /**
     * @brief       导出当前引擎全局物理 KV 缓存的底层数据到外部内存缓冲区中进行备份
     *
     * @param       参数名称: data_ptr                      数据类型:        void*
     *              参数说明: 外部接收导出的物理缓存数据的内存缓冲区指针。若传入 nullptr，则仅用于计算缓存所需的总大小并通过 out_size 返回。
     *
     * @param       参数名称: out_size                      数据类型:        size_t*
     *              参数说明: 传入时为外部缓冲区的最大承载容量限制，返回时被修改为实际写入或导出的物理大包所需要的实际字节数。
     *
     * @return      返回操作状态码：0 代表导出成功；其他非 0 数值则代表具体的失败错误码。
     *              数据类型: int
     **/
    DLL_EXPORT int LiteRtLm_GetKVCache(void* data_ptr, size_t* out_size);

    /**
     * @brief       恢复或批量同步重建底层的物理 KV 缓存
     *              可以从外部备份包恢复物理缓存，或者在不传数据包时触发一键批量预填充逻辑。
     *
     * @param       参数名称: data_ptr                      数据类型:        const void*
     *              参数说明: 指向之前备份的物理 KV 缓存数据大包的指针。若传入 nullptr 且 size 为 0，则触发“一键批量同步预填充”将所有已 Append 的逻辑历史同步为物理缓存。
     *
     * @param       参数名称: size                           数据类型:        size_t
     *              参数说明: 备份包数据的实际大小 (字节数)。
     *
     * @return      返回操作状态码：0 代表恢复/同步成功；其他非 0 数值则代表具体的失败错误码。
     *              数据类型: int
     **/
    DLL_EXPORT int LiteRtLm_SetKVCache(const void* data_ptr, size_t size);

#ifdef __cplusplus
}
#endif

#endif
