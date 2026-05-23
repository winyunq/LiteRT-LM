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
#include <windows.h>
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

#ifdef _MSC_VER
#include <sstream>
inline int FilterException(DWORD code, struct _EXCEPTION_POINTERS* ep) {
    std::stringstream ss;
    ss << "==================================================\n";
    ss << "[FATAL CRASHED] Captured SEH Exception 0x" << std::hex << code;
    if (ep && ep->ExceptionRecord) {
        ss << "\n  Exception Address (RIP): 0x" << ep->ExceptionRecord->ExceptionAddress;
        if (code == EXCEPTION_ACCESS_VIOLATION) {
            ss << "\n  Access Violation Detail:";
            ss << "\n    Access Operation: " << (ep->ExceptionRecord->ExceptionInformation[0] ? "WRITE" : "READ");
            ss << "\n    Target Address: 0x" << std::hex << ep->ExceptionRecord->ExceptionInformation[1];
        }
    }
    if (ep && ep->ContextRecord) {
        ss << "\n  CPU Register Context (x64):";
        ss << "\n    RIP: 0x" << std::hex << ep->ContextRecord->Rip;
        ss << "\n    RSP: 0x" << std::hex << ep->ContextRecord->Rsp;
        ss << "\n    RBP: 0x" << std::hex << ep->ContextRecord->Rbp;
        ss << "\n    RAX: 0x" << std::hex << ep->ContextRecord->Rax;
        ss << "\n    RBX: 0x" << std::hex << ep->ContextRecord->Rbx;
        ss << "\n    RCX: 0x" << std::hex << ep->ContextRecord->Rcx;
        ss << "\n    RDX: 0x" << std::hex << ep->ContextRecord->Rdx;
        ss << "\n    R8:  0x" << std::hex << ep->ContextRecord->R8;
        ss << "\n    R9:  0x" << std::hex << ep->ContextRecord->R9;
        ss << "\n    R10: 0x" << std::hex << ep->ContextRecord->R10;
        ss << "\n    R11: 0x" << std::hex << ep->ContextRecord->R11;
        ss << "\n    R12: 0x" << std::hex << ep->ContextRecord->R12;
    }
    ss << "\n==================================================";
    LogDebug(ss.str());
    return EXCEPTION_EXECUTE_HANDLER;
}

// 声明一个函数指针，用于在 SEH 保护下执行
typedef void (*SafeFunc)(void* arg);

// 纯 C 风格的物理防护层，没有任何 C++ 局部非平凡对象展开需求，彻底规避 C2712/C2713
static bool SafeExecuteWithSehGuard(SafeFunc func, void* arg) {
    __try {
        func(arg);
        return true;
    } __except (FilterException(GetExceptionCode(), GetExceptionInformation())) {
        return false;
    }
}
#endif

#include "runtime/conversation/conversation.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "absl/types/span.h"
#include "nlohmann/json.hpp"
#include "litert_lm_wrapper.h"

#include "runtime/components/constrained_decoding/llg_constraint_config.h"
#include "runtime/util/litert_util.h"

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

static Engine* CreateEngineAndConversationInternal(LiteRtLm_Config config, std::string& out_err) {
    LogDebug("[Trace] CreateEngineAndConversationInternal: Initiating core creation flow...");
    
    LogDebug("[Trace] Step 1: Parsing ModelAssets from path: " + (config.model_path ? std::string(config.model_path) : "NULL"));
    auto model_assets_or = ModelAssets::Create(config.model_path);
    if (!model_assets_or.ok()) {
        out_err = "ModelAssets::Create failed: " + std::string(model_assets_or.status().message());
        LogDebug("[Trace] Step 1 FAILED: " + out_err);
        return nullptr;
    }
    LogDebug("[Trace] Step 1 SUCCESS: ModelAssets created successfully.");
    
    LogDebug("[Trace] Step 2: Resolving backend from string: " + (config.backend ? std::string(config.backend) : "NULL"));
    auto backend_or = GetBackendFromString(config.backend);
    if (!backend_or.ok()) {
        out_err = "GetBackendFromString failed: " + std::string(backend_or.status().message());
        LogDebug("[Trace] Step 2 FAILED: " + out_err);
        return nullptr;
    }
    LogDebug("[Trace] Step 2 SUCCESS: Backend resolved.");

    // 根据开关决定是否启用多模态后端
    std::optional<Backend> vision_backend = config.bEnableVision ? std::make_optional(*backend_or) : std::nullopt;
    std::optional<Backend> audio_backend = config.bEnableAudio ? std::make_optional(*backend_or) : std::nullopt;
    LogDebug("[Trace] Modality States: bEnableVision=" + std::to_string(config.bEnableVision) + ", bEnableAudio=" + std::to_string(config.bEnableAudio));

    LogDebug("[Trace] Step 3: Generating EngineSettings::CreateDefault...");
    auto settings_or = EngineSettings::CreateDefault(
        std::move(*model_assets_or), 
        *backend_or, 
        vision_backend, 
        audio_backend
    );
    if (!settings_or.ok()) {
        out_err = "EngineSettings::CreateDefault failed: " + std::string(settings_or.status().message());
        LogDebug("[Trace] Step 3 FAILED: " + out_err);
        return nullptr;
    }
    LogDebug("[Trace] Step 3 SUCCESS: Default settings initialized.");

    auto& main_settings = settings_or->GetMutableMainExecutorSettings();

    if (config.max_num_tokens > 0) {
        main_settings.SetMaxNumTokens(config.max_num_tokens);
        LogDebug("[Trace] Set max_num_tokens = " + std::to_string(config.max_num_tokens));
    }
    
    // Step 3.3: CpuConfig —— 显式取出拷贝、修改并回写写回
    auto cpu_config_or = main_settings.MutableBackendConfig<CpuConfig>();
    if (cpu_config_or.ok()) {
        CpuConfig cpu_cfg = cpu_config_or.value();
        if (config.num_threads > 0) {
            cpu_cfg.number_of_threads = config.num_threads;
            LogDebug("[DIAGNOSTIC] Set CpuConfig number_of_threads = " + std::to_string(config.num_threads));
        }
        if (config.prefill_chunk_size > 0) {
            cpu_cfg.prefill_chunk_size = config.prefill_chunk_size;
            LogDebug("[DIAGNOSTIC] Set CpuConfig prefill_chunk_size = " + std::to_string(config.prefill_chunk_size));
        }
        main_settings.SetBackendConfig(cpu_cfg);
        LogDebug("[DIAGNOSTIC] Step 3.3 Finished. CpuConfig applied and written back.");
    } else {
        LogDebug("[DIAGNOSTIC] Step 3.3 Finished. CPU Config Status: " + std::string(cpu_config_or.status().message()));
    }

    // Step 3.4: GpuConfig —— GPU backend 下显式获取并回写
    auto gpu_config_or = main_settings.MutableBackendConfig<GpuConfig>();
    if (gpu_config_or.ok()) {
        GpuConfig gpu_cfg = gpu_config_or.value();
        main_settings.SetBackendConfig(gpu_cfg);
        LogDebug("[DIAGNOSTIC] Step 3.4 Finished. GpuConfig explicitly written back.");
    } else {
        LogDebug("[DIAGNOSTIC] Step 3.4 Finished. GPU Config Status: " + std::string(gpu_config_or.status().message()));
    }

    // Step 3.5: GpuArtisanConfig —— 获取、修改并回写
    auto gpu_artisan_config_or = main_settings.MutableBackendConfig<GpuArtisanConfig>();
    if (gpu_artisan_config_or.ok()) {
        GpuArtisanConfig gpu_cfg = gpu_artisan_config_or.value();
        gpu_cfg.set_enable_host_mapped_pointer = (config.bEnableHostMappedPointer != 0);
        main_settings.SetBackendConfig(gpu_cfg);
        LogDebug("[DIAGNOSTIC] Step 3.5 Finished. GpuArtisanConfig applied and written back: host_mapped=" + std::to_string(config.bEnableHostMappedPointer));
    } else {
        LogDebug("[DIAGNOSTIC] Step 3.5 Finished. GPU Artisan Config Status: " + std::string(gpu_artisan_config_or.status().message()));
    }

    // Step 3.6: AdvancedSettings —— 仅修改用户明确为 true 的字段，不覆盖默认值
    LogDebug("[DIAGNOSTIC] Step 3.6: Updating AdvancedSettings...");
    AdvancedSettings adv = main_settings.GetAdvancedSettings().value_or(AdvancedSettings());
    if (config.bOptimizeShader) {
        adv.optimize_shader_compilation = true;
    }
    if (config.bEnableBenchmark) {
        adv.is_benchmark = true;
    }
    if (config.bShareConstantTensors) {
        adv.share_constant_tensors = true;
    }
    main_settings.SetAdvancedSettings(adv);
    LogDebug("[DIAGNOSTIC] Step 3.6 Finished.");

    // Step 3.7: 与 litert_lm_main.cc 对齐，显式初始化 BenchmarkParams 避免内部 nullptr
    LogDebug("[DIAGNOSTIC] Step 3.7: Initializing BenchmarkParams...");
    if (config.bEnableBenchmark) {
        settings_or->GetMutableBenchmarkParams() = proto::BenchmarkParams();
        LogDebug("[DIAGNOSTIC] Step 3.7: BenchmarkParams initialized (benchmark mode).");
    }
    LogDebug("[DIAGNOSTIC] Step 3.7 Finished.");
    
    LogDebug("[Trace] Step 4: Calling EngineFactory::CreateDefault (Crucial backend init point)...");
    auto engine_or = EngineFactory::CreateDefault(std::move(*settings_or));
    if (!engine_or.ok()) {
        out_err = "EngineFactory::CreateDefault failed: " + std::string(engine_or.status().message());
        LogDebug("[Trace] Step 4 FAILED: " + out_err);
        return nullptr;
    }
    LogDebug("[Trace] Step 4 SUCCESS: Engine instance created.");

    Engine* engine = engine_or->release();

    // --- 隐式自动初始化全局唯一的会话上下文 ---
    LogDebug("[Trace] Step 5: Preparing ConversationConfig and session mapping...");
    auto builder = ConversationConfig::Builder();
    builder.SetEnableConstrainedDecoding(false);
    builder.SetConstraintProviderConfig(LlGuidanceConfig());

    if (config.tools_json && strlen(config.tools_json) > 0) {
        try {
            JsonPreface preface;
            preface.tools = nlohmann::ordered_json::parse(config.tools_json);
            builder.SetPreface(preface);
            LogDebug("[Trace] Preface tools successfully parsed and set.");
        } catch (const std::exception& e) {
            LogDebug("[Trace] WARNING - Parse tools_json failed: " + std::string(e.what()));
        }
    }

    // 创建默认的会话配置
    auto sess_cfg = SessionConfig::CreateDefault();

    if (engine->GetEngineSettings().GetVisionExecutorSettings().has_value()) {
        sess_cfg.SetVisionModalityEnabled(true);
        LogDebug("[Trace] Vision modality detected and enabled in SessionConfig.");
    }

    if (engine->GetEngineSettings().GetAudioExecutorSettings().has_value()) {
        sess_cfg.SetAudioModalityEnabled(true);
        LogDebug("[Trace] Audio modality detected and enabled in SessionConfig.");
    }

    LogDebug("[Trace] Step 6: Building ConversationConfig...");
    auto conv_cfg_or = builder.SetSessionConfig(sess_cfg).Build(*engine);
    if (!conv_cfg_or.ok()) {
        out_err = "Build ConversationConfig failed: " + std::string(conv_cfg_or.status().message());
        LogDebug("[Trace] Step 6 FAILED: " + out_err);
        delete engine;
        return nullptr;
    }
    LogDebug("[Trace] Step 6 SUCCESS: ConversationConfig built.");

    LogDebug("[Trace] Step 7: Creating Conversation instance...");
    auto conv_or = Conversation::Create(*engine, *conv_cfg_or);
    if (!conv_or.ok()) {
        out_err = "Conversation::Create failed: " + std::string(conv_or.status().message());
        LogDebug("[Trace] Step 7 FAILED: " + out_err);
        delete engine;
        return nullptr;
    }
    LogDebug("[Trace] Step 7 SUCCESS: Conversation created successfully.");

    g_GlobalConversationContext = std::make_unique<LiteRtLm_ConversationContext>();
    g_GlobalConversationContext->conversation = std::move(*conv_or);

    LogDebug("[Trace] CreateEngineAndConversationInternal: Full flow completed SUCCESS.");
    return engine;
}

struct CreateEngineParam {
    LiteRtLm_Config config;
    std::string* out_err;
    Engine* out_engine;
};

static void RunEngineCreate(void* arg) {
    CreateEngineParam* param = static_cast<CreateEngineParam*>(arg);
    param->out_engine = CreateEngineAndConversationInternal(param->config, *(param->out_err));
}

DLL_EXPORT void* LiteRtLm_CreateEngine(LiteRtLm_Config config) {
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    LogDebug("==================================================");
    LogDebug("LiteRtLm_CreateEngine: Initiating safe creation...");
    LogDebug("  [Config] model_path: " + (config.model_path ? std::string(config.model_path) : "NULL"));
    LogDebug("  [Config] backend: " + (config.backend ? std::string(config.backend) : "NULL"));
    LogDebug("  [Config] max_num_tokens: " + std::to_string(config.max_num_tokens));
    LogDebug("  [Config] prefill_chunk_size: " + std::to_string(config.prefill_chunk_size));
    LogDebug("  [Config] num_threads: " + std::to_string(config.num_threads));
    LogDebug("  [Config] bOptimizeShader: " + std::to_string(config.bOptimizeShader));
    LogDebug("  [Config] bEnableBenchmark: " + std::to_string(config.bEnableBenchmark));
    LogDebug("  [Config] bShareConstantTensors: " + std::to_string(config.bShareConstantTensors));
    LogDebug("  [Config] bEnableHostMappedPointer: " + std::to_string(config.bEnableHostMappedPointer));
    LogDebug("==================================================");

    std::string err_buf;
    CreateEngineParam param = { config, &err_buf, nullptr };

#ifdef _MSC_VER
    LogDebug("[Trace] Invoking SafeExecuteWithSehGuard...");
    bool ok = SafeExecuteWithSehGuard(RunEngineCreate, &param);
    if (!ok) {
        LogDebug("[Trace] SafeExecuteWithSehGuard captured a SEH hard exception inside creation execution!");
        g_ActiveEnginePtr = nullptr;
        return nullptr;
    }
#else
    RunEngineCreate(&param);
#endif

    if (!param.out_engine) {
        LogDebug("[Trace] Engine creation returned nullptr. Diagnostics: " + err_buf);
        g_ActiveEnginePtr = nullptr;
        return nullptr;
    }

    g_ActiveEnginePtr = static_cast<void*>(param.out_engine);
    LogDebug("[Trace] LiteRtLm_CreateEngine completed successfully. Engine handle stored: " + std::to_string((uintptr_t)g_ActiveEnginePtr));
    return g_ActiveEnginePtr;
}

DLL_EXPORT void LiteRtLm_DestroyEngine(void* engine_ptr) {
    std::lock_guard<std::mutex> global_lock(g_GlobalContextMutex);
    LogDebug("LiteRtLm_DestroyEngine: Destroying active engine and conversation...");
    
    g_GlobalConversationContext.reset(); // 1. 自动隐式销毁全局会话
    
    if (engine_ptr) {
        delete static_cast<Engine*>(engine_ptr); // 2. 销毁引擎实例
    }
    if (g_ActiveEnginePtr == engine_ptr) {
        g_ActiveEnginePtr = nullptr;
    }
    
    // 3. 在引擎和会话彻底安全销毁后，物理强力清空 LiteRT 全局运行环境缓存，强制触发底层 WebGPU 析构并彻底退还显存！
    litert::lm::ClearCachedEnvironments();
    
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
            
            // [绝杀修复] 从全局会话的历史中提取刚完成生成的完美、完整 Message
            try {
                auto history = ctx->conversation->GetHistory();
                if (!history.empty()) {
                    const auto& final_msg = history.back();
                    ctx->cb_json_buffer = final_msg.dump();
                    res.full_json_chunk = ctx->cb_json_buffer.c_str();
                    LogDebug("LiteRtLm_RunInference Callback Done: Fetched final message from history: " + ctx->cb_json_buffer);
                }
            } catch (const std::exception& e) {
                LogDebug("LiteRtLm_RunInference Callback Done: Fetch final message failed: " + std::string(e.what()));
            }

            res.bIsDone = 1;
            callback(res, user_ptr); 
            return;
        }

        ctx->cb_json_buffer = chunk->dump();
        if (chunk->contains("tool_calls")) {
            res.full_json_chunk = ctx->cb_json_buffer.c_str();
        } else {
            res.full_json_chunk = nullptr;
        }

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

    auto cache_or = session->GetKVCache();
    if (!cache_or.ok()) {
        LogDebug("LiteRtLm_GetKVCache: FAILED - GetKVCache failed: " + std::string(cache_or.status().message()));
        return -4;
    }

    const std::string& cache_data = cache_or.value();
    size_t required_size = cache_data.size();

    if (!data_ptr) {
        *out_size = required_size;
        return 0;
    }

    if (*out_size < required_size) {
        LogDebug("LiteRtLm_GetKVCache: FAILED - Provided data buffer is too small.");
        *out_size = required_size;
        return -7;
    }

    std::memcpy(data_ptr, cache_data.data(), required_size);
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

    // A. 擦除/重置逻辑
    if (!data_ptr && size == 0) {
        auto status = session->SetKVCache("");
        if (!status.ok()) {
            LogDebug("LiteRtLm_SetKVCache: Reset active KV cache FAILED: " + std::string(status.message()));
            return -5;
        }
        LogDebug("LiteRtLm_SetKVCache: Successfully reset and cleared active KV cache.");
        return 0;
    }

    // B. 灌入逻辑
    absl::string_view cache_data(static_cast<const char*>(data_ptr), size);
    auto status = session->SetKVCache(cache_data);
    if (!status.ok()) {
        LogDebug("LiteRtLm_SetKVCache: FAILED - SetKVCache failed: " + std::string(status.message()));
        return -6;
    }

    LogDebug("LiteRtLm_SetKVCache: Successfully restored physical KV cache into GPU memory.");
    return 0;
}

}
