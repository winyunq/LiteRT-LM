// Copyright (c) 2025-2026 Winyunq. All rights reserved.
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <algorithm>

#ifdef _MSC_VER
#pragma comment(linker, "/EXPORT:LiteRtLm_CreateEngine")
#pragma comment(linker, "/EXPORT:LiteRtLm_DestroyEngine")
#pragma comment(linker, "/EXPORT:LiteRtLm_AppendUserMessage")
#pragma comment(linker, "/EXPORT:LiteRtLm_AppendAssistantMessage")
#pragma comment(linker, "/EXPORT:LiteRtLm_RunInference")
#pragma comment(linker, "/EXPORT:LiteRtLm_StopMessage")
#pragma comment(linker, "/EXPORT:LiteRtLm_WaitUntilDone")
#pragma comment(linker, "/EXPORT:LiteRtLm_GetKVCache")
#pragma comment(linker, "/EXPORT:LiteRtLm_SetKVCache")
#endif

inline void LogDebug(const std::string& msg) {
    try {
        std::ofstream log_file("D:\\LiteRT-LM\\WinyunqDebug\\litert_lm_wrapper_debug.log", std::ios::app);
        if (log_file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t_now = std::chrono::system_clock::to_time_t(now);
            log_file << "[" << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << "] " << msg << std::endl;
        }
    } catch (...) {}
}

#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/core/session_basic.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "absl/types/span.h"
#include "nlohmann/json.hpp"
#include "litert_lm_wrapper.h"

#include "runtime/components/constrained_decoding/llg_constraint_config.h"

using namespace litert::lm;
using json = nlohmann::json;

/**
 * 内部状态容器，确保 Wrapper 能像 CLI 一样管理全局唯一的对话
 */
struct LiteRtLm_ConversationContext {
    std::unique_ptr<Conversation> conversation;
    std::string pending_json_msg; // 缓存多模态 JSON 消息
    std::mutex mtx;

    // 持久化回调字符串缓冲区，彻底消除跨 DLL 异步 lambda 野指针悬空崩溃 (Jun 2026)
    std::string cb_text_buffer;
    std::string cb_json_buffer;
    std::string cb_error_buffer;
};

// 全局唯一的会话状态单例
static void* g_ActiveEnginePtr = nullptr;
static std::unique_ptr<LiteRtLm_ConversationContext> g_GlobalConversationContext = nullptr;
static std::mutex g_GlobalContextMutex;

extern "C" {

DLL_EXPORT const char* LiteRtLm_GetAvailableBackends() {
    return "cpu,gpu"; 
}

DLL_EXPORT void* LiteRtLm_CreateEngine(LiteRtLm_Config config) {
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    LogDebug("LiteRtLm_CreateEngine: Starting engine creation...");

    auto model_assets_or = ModelAssets::Create(config.model_path);
    if (!model_assets_or.ok()) {
        LogDebug("LiteRtLm_CreateEngine: ModelAssets::Create failed: " + std::string(model_assets_or.status().message()));
        return nullptr;
    }
    
    auto backend_or = GetBackendFromString(config.backend);
    if (!backend_or.ok()) {
        LogDebug("LiteRtLm_CreateEngine: GetBackendFromString failed: " + std::string(backend_or.status().message()));
        return nullptr;
    }

    // 根据开关决定是否启用多模态后端
    std::optional<Backend> vision_backend = config.bEnableVision ? std::make_optional(*backend_or) : std::nullopt;
    std::optional<Backend> audio_backend = config.bEnableAudio ? std::make_optional(*backend_or) : std::nullopt;

    auto settings_or = EngineSettings::CreateDefault(
        std::move(*model_assets_or), 
        *backend_or, 
        vision_backend, 
        audio_backend
    );
    if (!settings_or.ok()) {
        LogDebug("LiteRtLm_CreateEngine: CreateDefault settings failed: " + std::string(settings_or.status().message()));
        return nullptr;
    }

    auto& main_settings = settings_or->GetMutableMainExecutorSettings();

    if (config.max_num_tokens > 0) {
        main_settings.SetMaxNumTokens(config.max_num_tokens);
    }
    auto cpu_config_or = main_settings.MutableBackendConfig<CpuConfig>();
    if (cpu_config_or.ok()) {
        if (config.num_threads > 0) {
            cpu_config_or->number_of_threads = config.num_threads;
        }
        if (config.prefill_chunk_size > 0) {
            cpu_config_or->prefill_chunk_size = config.prefill_chunk_size;
            LogDebug("LiteRtLm_CreateEngine: Set prefill_chunk_size limit to: " + std::to_string(config.prefill_chunk_size));
        }
    }

    AdvancedSettings adv = main_settings.GetAdvancedSettings().value_or(AdvancedSettings());
    adv.optimize_shader_compilation = (config.bOptimizeShader != 0);
    adv.is_benchmark = (config.bEnableBenchmark != 0);
    main_settings.SetAdvancedSettings(adv);
    
    auto engine_or = EngineFactory::CreateAny(std::move(*settings_or));
    if (!engine_or.ok()) {
        LogDebug("LiteRtLm_CreateEngine: CreateAny engine failed: " + std::string(engine_or.status().message()));
        return nullptr;
    }

    Engine* engine = engine_or->release();
    g_ActiveEnginePtr = static_cast<void*>(engine);

    // --- 隐式自动初始化全局唯一的会话上下文 ---
    auto builder = ConversationConfig::Builder();
    builder.SetEnableConstrainedDecoding(false);
    builder.SetConstraintProviderConfig(LlGuidanceConfig());

    // 创建默认的会话配置
    auto sess_cfg = SessionConfig::CreateDefault();

    // 探测引擎多模态支持状态，若视觉执行器就绪则自动启用视觉模态
    if (engine->GetVisionExecutorProperties().ok()) {
        sess_cfg.SetVisionModalityEnabled(true);
        LogDebug("LiteRtLm_CreateEngine: Vision modality dynamically enabled in SessionConfig.");
    }

    // 探测引擎多模态支持状态，若音频执行器就绪则自动启用音频模态
    if (engine->GetAudioExecutorProperties().ok()) {
        sess_cfg.SetAudioModalityEnabled(true);
        LogDebug("LiteRtLm_CreateEngine: Audio modality dynamically enabled in SessionConfig.");
    }

    // 构建并生成最终的会话配置
    auto conv_cfg_or = builder.SetSessionConfig(sess_cfg).Build(*engine);
    if (!conv_cfg_or.ok()) {
        LogDebug("LiteRtLm_CreateEngine: Build ConversationConfig failed: " + std::string(conv_cfg_or.status().message()));
        delete engine;
        g_ActiveEnginePtr = nullptr;
        return nullptr;
    }

    auto conv_or = Conversation::Create(*engine, *conv_cfg_or);
    if (!conv_or.ok()) {
        LogDebug("LiteRtLm_CreateEngine: Conversation::Create failed: " + std::string(conv_or.status().message()));
        delete engine;
        g_ActiveEnginePtr = nullptr;
        return nullptr;
    }

    g_GlobalConversationContext = std::make_unique<LiteRtLm_ConversationContext>();
    g_GlobalConversationContext->conversation = std::move(*conv_or);

    LogDebug("LiteRtLm_CreateEngine: Successfully created engine and active single conversation instance.");
    return g_ActiveEnginePtr;
}

DLL_EXPORT void LiteRtLm_DestroyEngine(void* engine_ptr) {
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    LogDebug("LiteRtLm_DestroyEngine: Destroying active engine and conversation...");
    
    g_GlobalConversationContext.reset(); // 自动隐式销毁全局会话
    if (engine_ptr) {
        delete static_cast<Engine*>(engine_ptr);
    }
    if (g_ActiveEnginePtr == engine_ptr) {
        g_ActiveEnginePtr = nullptr;
    }
    LogDebug("LiteRtLm_DestroyEngine: Destroy complete.");
}

DLL_EXPORT void LiteRtLm_AppendUserMessage(const char* json_msg) {
    if (!json_msg) return;
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    if (!g_GlobalConversationContext) {
        LogDebug("LiteRtLm_AppendUserMessage: FAILED - No active conversation context.");
        return;
    }
    auto* ctx = g_GlobalConversationContext.get();
    std::lock_guard<std::mutex> lock(ctx->mtx);
    ctx->pending_json_msg = json_msg;
    LogDebug("LiteRtLm_AppendUserMessage cached pending JSON: " + std::string(json_msg));
}

DLL_EXPORT void LiteRtLm_AppendAssistantMessage(const char* text) {
    if (!text) return;
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    if (!g_GlobalConversationContext) return;
    auto* ctx = g_GlobalConversationContext.get();
    json msg = {{"role", "assistant"}, {"content", {{{"type", "text"}, {"text", text}}}}};
    OptionalArgs args;
    args.has_pending_message = true;
    ctx->conversation->SendMessage(msg, std::move(args));
}

DLL_EXPORT void LiteRtLm_RunInference(LiteRtLm_SamplingParams params, LiteRtLmCallback callback, void* user_ptr) {
    if (!callback) return;
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    if (!g_GlobalConversationContext) {
        LogDebug("LiteRtLm_RunInference: FAILED - No active conversation context.");
        LiteRtLm_Result res = {nullptr, nullptr, "No active conversation context.", 1, 0.0f};
        callback(res, user_ptr);
        return;
    }
    auto* ctx = g_GlobalConversationContext.get();

    // 构建触发消息
    json msg_to_send;
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        if (!ctx->pending_json_msg.empty()) {
            try {
                msg_to_send = json::parse(ctx->pending_json_msg);
                LogDebug("LiteRtLm_RunInference: Parsed pending JSON.");
                /// 若底层的视觉引擎未被创建（视觉模态被停用），清洗过滤 JSON 中的 image 资源以平滑降级为纯文本交互，防范底层执行器空指针崩溃
                if (!ctx->conversation->GetConfig().GetSessionConfig().VisionModalityEnabled()) {
                    if (msg_to_send.contains("content")) {
                        auto& content = msg_to_send["content"];
                        if (content.is_array()) {
                            json filtered_content = json::array();
                            for (const auto& item : content) {
                                if (item.is_object() && item.contains("type") && item["type"] == "image") {
                                    LogDebug("LiteRtLm_RunInference: Warning - Filtering out 'image' item because Vision Modality is disabled.");
                                    continue;
                                }
                                filtered_content.push_back(item);
                            }
                            content = filtered_content.empty() ? json("") : filtered_content;
                        } else if (content.is_object()) {
                            if (content.contains("type") && content["type"] == "image") {
                                LogDebug("LiteRtLm_RunInference: Warning - Filtering out single 'image' content object because Vision Modality is disabled.");
                                content = "";
                            }
                        }
                    }
                }
            } catch (const std::exception& e) {
                LogDebug("LiteRtLm_RunInference: Parse JSON error: " + std::string(e.what()));
                msg_to_send = json::object({
                    {"role", "user"},
                    {"content", {{{"type", "text"}, {"text", "Please describe this image."}}}}
                });
            }
            ctx->pending_json_msg.clear();
        } else {
            msg_to_send = json::object({
                {"role", "user"},
                {"content", ""}
            });
            LogDebug("LiteRtLm_RunInference: Triggering pending message.");
        }
    }

    OptionalArgs args;
    args.max_output_tokens = params.max_tokens > 0 ? std::make_optional(params.max_tokens) : std::nullopt;
    
    if (params.constraint_type > 0 && params.constraint_string) {
        LlGuidanceConstraintArg llg_arg;
        switch (params.constraint_type) {
            case 1: llg_arg.constraint_type = LlgConstraintType::kRegex; break;
            case 2: llg_arg.constraint_type = LlgConstraintType::kJsonSchema; break;
            case 3: llg_arg.constraint_type = LlgConstraintType::kLark; break;
        }
        llg_arg.constraint_string = params.constraint_string;
        args.decoding_constraint = llg_arg;
        LogDebug("LiteRtLm_RunInference: Constrained decoding enabled.");
    }

    // 内部 Lambda 包装回调，确保线程安全和字符串存活
    auto internal_cb = [ctx, callback, user_ptr](absl::StatusOr<Message> chunk) {
        LiteRtLm_Result res = {nullptr, nullptr, nullptr, 0, 0.0f};
        std::lock_guard<std::mutex> lock(ctx->mtx);

        if (!chunk.ok()) {
            ctx->cb_error_buffer = chunk.status().message();
            LogDebug("LiteRtLm_RunInference Callback: Status Error - " + ctx->cb_error_buffer);
            res.error_msg = ctx->cb_error_buffer.c_str(); 
            res.bIsDone = 1;
            callback(res, user_ptr); 
            return;
        }
        
        if (chunk->is_null() || chunk->empty()) {
            LogDebug("LiteRtLm_RunInference Callback: Generation Done (empty chunk).");
            res.bIsDone = 1;
            callback(res, user_ptr); 
            return;
        }

        ctx->cb_json_buffer = chunk->dump();
        res.full_json_chunk = ctx->cb_json_buffer.c_str();

        ctx->cb_text_buffer.clear();
        if (chunk->contains("content")) {
            auto& content = (*chunk)["content"];
            if (content.is_array()) {
                for (const auto& part : content) {
                    if (part.is_object() && part.contains("text")) {
                        ctx->cb_text_buffer += part["text"].get<std::string>();
                    }
                }
            }
        }
        
        res.text_chunk = ctx->cb_text_buffer.empty() ? nullptr : ctx->cb_text_buffer.c_str();
        res.bIsDone = 0; 

        callback(res, user_ptr);
    };

    LogDebug("LiteRtLm_RunInference: Calling SendMessageAsync...");
    auto status = ctx->conversation->SendMessageAsync(msg_to_send, std::move(internal_cb), std::move(args));
    if (!status.ok()) {
        LogDebug("LiteRtLm_RunInference: SendMessageAsync FAILED: " + std::string(status.message()));
    } else {
        LogDebug("LiteRtLm_RunInference: SendMessageAsync accepted.");
    }
}

DLL_EXPORT void LiteRtLm_StopMessage() {
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    if (g_GlobalConversationContext) {
        g_GlobalConversationContext->conversation->CancelProcess();
    }
}

DLL_EXPORT int LiteRtLm_WaitUntilDone(void* engine_ptr, int timeout_sec) {
    if (!engine_ptr) return -1;
    Engine* engine = static_cast<Engine*>(engine_ptr);
    absl::Duration timeout = timeout_sec > 0 
        ? absl::Seconds(timeout_sec) 
        : Engine::kDefaultTimeout;
    auto status = engine->WaitUntilDone(timeout);
    if (status.ok()) return 0;
    if (absl::IsDeadlineExceeded(status)) return 1;
    return -1;
}

// --- 2. GPU 物理 KV 缓存直接锁显存导出与恢复接口 ---

DLL_EXPORT int LiteRtLm_GetKVCache(void* data_ptr, size_t* out_size) {
    if (!out_size) return -1;
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    
    if (!g_GlobalConversationContext) {
        LogDebug("LiteRtLm_GetKVCache: FAILED - No active conversation context.");
        return -2;
    }

    auto* session = g_GlobalConversationContext->conversation->GetSession();
    if (!session) {
        LogDebug("LiteRtLm_GetKVCache: FAILED - No active session.");
        return -3;
    }

    auto* executor = session->GetLlmExecutor();
    if (!executor) {
        LogDebug("LiteRtLm_GetKVCache: FAILED - No active LlmExecutor.");
        return -4;
    }

    auto* compiled_exec = dynamic_cast<LlmLiteRtCompiledModelExecutorBase*>(executor);
    if (!compiled_exec) {
        compiled_exec = static_cast<LlmLiteRtCompiledModelExecutorBase*>(executor);
    }

    auto& map1 = compiled_exec->GetKVCacheBuffers1();
    auto& map2 = compiled_exec->GetKVCacheBuffers2();

    // 收集所有物理缓存 tensors 并且排好序以确物理顺序绝对一致
    std::vector<std::pair<std::string, ::litert::TensorBuffer*>> all_tensors;
    for (auto& pair : map1) {
        all_tensors.push_back({std::string(pair.first), &pair.second});
    }
    for (auto& pair : map2) {
        all_tensors.push_back({"_map2_" + std::string(pair.first), &pair.second});
    }

    std::sort(all_tensors.begin(), all_tensors.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // 物理序列化二进制数据流
    std::vector<char> stream;
    
    // 1. 包头：Magic Number + Version + Count
    uint32_t magic = 0x594E515F; // "WNYQ"
    uint32_t version = 1;
    uint32_t count = static_cast<uint32_t>(all_tensors.size());

    auto append_val = [&stream](const void* val_ptr, size_t val_size) {
        const char* byte_ptr = static_cast<const char*>(val_ptr);
        stream.insert(stream.end(), byte_ptr, byte_ptr + val_size);
    };

    append_val(&magic, sizeof(magic));
    append_val(&version, sizeof(version));
    append_val(&count, sizeof(count));

    // 2. 依次读取并打包每一个 TensorBuffer 的数据
    for (auto& pair : all_tensors) {
        const std::string& name = pair.first;
        auto* buffer = pair.second;

        auto size_expected = buffer->PackedSize();
        if (!size_expected.HasValue()) {
            LogDebug("LiteRtLm_GetKVCache: FAILED - Failed to get PackedSize for tensor: " + name);
            return -5;
        }
        size_t tensor_bytes = size_expected.Value();

        uint32_t name_len = static_cast<uint32_t>(name.size());
        append_val(&name_len, sizeof(name_len));
        stream.insert(stream.end(), name.begin(), name.end());

        uint32_t data_len = static_cast<uint32_t>(tensor_bytes);
        append_val(&data_len, sizeof(data_len));

        // 物理分配临时接收缓冲区直接锁显存读取物理字节
        std::vector<char> temp_data(tensor_bytes);
        auto read_expected = buffer->Read<char>(absl::MakeSpan(temp_data));
        if (!read_expected.HasValue()) {
            LogDebug("LiteRtLm_GetKVCache: FAILED - Failed to read physical data from GPU for: " + name);
            return -6;
        }

        stream.insert(stream.end(), temp_data.begin(), temp_data.end());
    }

    // 3. 返回大小或执行实际物理拷贝到外部缓冲区
    size_t required_size = stream.size();
    if (!data_ptr) {
        *out_size = required_size;
        return 0;
    }

    if (*out_size < required_size) {
        LogDebug("LiteRtLm_GetKVCache: FAILED - Provided data buffer is too small.");
        *out_size = required_size;
        return -7;
    }

    std::memcpy(data_ptr, stream.data(), required_size);
    *out_size = required_size;
    
    LogDebug("LiteRtLm_GetKVCache: Successfully exported physical KV cache, size: " + std::to_string(required_size) + " bytes.");
    return 0;
}

DLL_EXPORT int LiteRtLm_SetKVCache(const void* data_ptr, size_t size) {
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    
    if (!g_GlobalConversationContext) {
        LogDebug("LiteRtLm_SetKVCache: FAILED - No active conversation context.");
        return -2;
    }

    auto* session = g_GlobalConversationContext->conversation->GetSession();
    if (!session) {
        LogDebug("LiteRtLm_SetKVCache: FAILED - No active session.");
        return -3;
    }

    auto* executor = session->GetLlmExecutor();
    if (!executor) {
        LogDebug("LiteRtLm_SetKVCache: FAILED - No active LlmExecutor.");
        return -4;
    }

    // A. 擦除/重置逻辑
    if (!data_ptr && size == 0) {
        auto status = executor->Reset();
        if (!status.ok()) {
            LogDebug("LiteRtLm_SetKVCache: Reset active KV cache FAILED: " + std::string(status.message()));
            return -5;
        }
        LogDebug("LiteRtLm_SetKVCache: Successfully reset and cleared active KV cache.");
        return 0;
    }

    // B. 反序列化物理灌入显存
    auto* compiled_exec = dynamic_cast<LlmLiteRtCompiledModelExecutorBase*>(executor);
    if (!compiled_exec) {
        compiled_exec = static_cast<LlmLiteRtCompiledModelExecutorBase*>(executor);
    }

    auto& map1 = compiled_exec->GetKVCacheBuffers1();
    auto& map2 = compiled_exec->GetKVCacheBuffers2();

    const char* stream_ptr = static_cast<const char*>(data_ptr);
    size_t offset = 0;

    auto read_val = [&stream_ptr, &offset, size](void* dest, size_t val_size) -> bool {
        if (offset + val_size > size) return false;
        std::memcpy(dest, stream_ptr + offset, val_size);
        offset += val_size;
        return true;
    };

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t count = 0;

    if (!read_val(&magic, sizeof(magic)) || magic != 0x594E515F) {
        LogDebug("LiteRtLm_SetKVCache: FAILED - Invalid magic number.");
        return -6;
    }
    if (!read_val(&version, sizeof(version)) || version != 1) {
        LogDebug("LiteRtLm_SetKVCache: FAILED - Unsupported packet version.");
        return -7;
    }
    if (!read_val(&count, sizeof(count))) {
        LogDebug("LiteRtLm_SetKVCache: FAILED - Failed to read count.");
        return -8;
    }

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t name_len = 0;
        if (!read_val(&name_len, sizeof(name_len))) return -9;

        if (offset + name_len > size) return -10;
        std::string name(stream_ptr + offset, name_len);
        offset += name_len;

        uint32_t data_len = 0;
        if (!read_val(&data_len, sizeof(data_len))) return -11;

        if (offset + data_len > size) return -12;
        const char* raw_tensor_data = stream_ptr + offset;
        offset += data_len;

        // 根据名称的前缀找到具体属于 map1 还是 map2
        bool is_map2 = (name.rfind("_map2_", 0) == 0);
        std::string origin_name = is_map2 ? name.substr(6) : name;

        auto& target_map = is_map2 ? map2 : map1;
        auto it = target_map.find(origin_name);
        if (it != target_map.end()) {
            auto& buffer = it->second;
            // 物理锁显存将数据直接灌写回 GPU 显存
            auto write_expected = buffer.Write<char>(absl::MakeSpan(raw_tensor_data, data_len));
            if (!write_expected.HasValue()) {
                LogDebug("LiteRtLm_SetKVCache: FAILED - Failed to write physical data to GPU for: " + name);
                return -13;
            }
        } else {
            LogDebug("LiteRtLm_SetKVCache: WARNING - Tensor key not found in active executors, skipping: " + origin_name);
        }
    }

    LogDebug("LiteRtLm_SetKVCache: Successfully restored physical KV cache into GPU memory.");
    return 0;
}

}
