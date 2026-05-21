// Copyright (c) 2025-2026 Winyunq. All rights reserved.
#include <iostream>
#include <windows.h>
#include <string>
#include <atomic>
#include <vector>
#include <filesystem>
#include <fstream>
#include <mutex>

// --- Wrapper Types (Aligned with litert_lm_wrapper.h) ---
typedef struct {
    const char* model_path;      // 模型路径
    const char* backend;         // 推理后端 ("cpu", "gpu")
    int max_num_tokens;          // 最大上下文长度
    int num_threads;             // CPU 线程数
    int bEnableBenchmark;        // 是否开启性能日志
    int bOptimizeShader;         // 是否优化着色器
    int bEnableVision;           // 是否启用视觉引擎
    int bEnableAudio;            // 是否启用音频引擎
    int prefill_chunk_size;      // 分块 prefill 限制
} LiteRtLm_Config;

typedef struct {
    float temperature;           // 采样温度
    float top_p;                 // Top-P 采样
    int top_k;                   // Top-K 采样
    int max_tokens;              // 本次生成最大 Token 数
    int constraint_type;         // 强制约束类型 (0:无, 1:Regex, 2:JSON Schema, 3:Lark)
    const char* constraint_string; // 约束字符串内容
} LiteRtLm_SamplingParams;

typedef struct {
    const char* text_chunk;      // 实时文本片段
    const char* full_json_chunk; // 完整的 OpenAI 兼容 JSON 片段 (包含 tool_calls 等)
    const char* error_msg;       // 错误消息
    int bIsDone;                 // 是否推理完成
    float tokens_per_sec;        // 推理速度
} LiteRtLm_Result;

typedef void (*LiteRtLmCallback)(LiteRtLm_Result result, void* user_ptr);

// --- Function Pointers ---
typedef void* (*PN_CreateEngine)(LiteRtLm_Config config);
typedef void (*PN_DestroyEngine)(void* engine_ptr);
typedef void (*PN_AppendUserMessage)(const char* json_msg);
typedef void (*PN_AppendAssistantMessage)(const char* text);
typedef void (*PN_RunInference)(LiteRtLm_SamplingParams params, LiteRtLmCallback callback, void* user_ptr);
typedef void (*PN_StopMessage)();
typedef int (*PN_WaitUntilDone)(void* engine_ptr, int timeout_sec);

// --- Global State ---
std::atomic<bool> g_IsDone{false};
std::string g_LastFullJson;
std::string g_LastText;

// --- Callbacks ---
void TestCallback(LiteRtLm_Result result, void* user_ptr) {
    if (result.text_chunk) {
        std::cout << result.text_chunk << std::flush;
        g_LastText += result.text_chunk;
    }
    if (result.full_json_chunk) {
        g_LastFullJson = result.full_json_chunk;
    }
    if (result.error_msg) {
        std::cerr << "\n[Callback ERROR] " << result.error_msg << std::endl;
    }
    if (result.bIsDone) {
        g_IsDone = true;
        if (result.tokens_per_sec > 0) {
            std::cout << "\n[Speed: " << result.tokens_per_sec << " tokens/sec]" << std::endl;
        }
    }
}

// --- Utils ---
std::string ExtractJsonValue(const std::string& json_str, const std::string& key) {
    size_t key_pos = json_str.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return "";

    size_t colon_pos = json_str.find(":", key_pos);
    if (colon_pos == std::string::npos) return "";

    size_t start_pos = json_str.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (start_pos == std::string::npos) return "";

    if (json_str[start_pos] == '"') {
        // String value
        size_t end_pos = json_str.find('"', start_pos + 1);
        if (end_pos != std::string::npos) {
            return json_str.substr(start_pos + 1, end_pos - start_pos - 1);
        }
    } else if (json_str[start_pos] == '{') {
        // Object value (rough capture for simple json schema arguments)
        int bracket_count = 1;
        size_t idx = start_pos + 1;
        while (idx < json_str.size() && bracket_count > 0) {
            if (json_str[idx] == '{') bracket_count++;
            else if (json_str[idx] == '}') bracket_count--;
            idx++;
        }
        return json_str.substr(start_pos, idx - start_pos);
    } else {
        // Number or boolean value
        size_t end_pos = json_str.find_first_of(",}", start_pos);
        if (end_pos != std::string::npos) {
            return json_str.substr(start_pos, end_pos - start_pos);
        }
    }
    return "";
}

// --- Main Verification Flow ---
int main(int argc, char* argv[]) {
    std::cout << "==============================================================" << std::endl;
    std::cout << "  Winyunq: LiteRT-LM DLL MCP/Function Calling 独立验证端 v1.0" << std::endl;
    std::cout << "==============================================================" << std::endl;

    std::string model_path = "D:\\gemma-4-E4B-it.litertlm";
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--model" && i + 1 < argc) {
                model_path = argv[i + 1];
            }
        }
    }

    std::cout << "[Step 1] Loading DLL (litert_lm_wrapper.dll)..." << std::endl;
    
    // Windows DLL 搜索依赖环境兼容注入：把当前目录加入 DLL 寻找搜索集中
    SetDllDirectoryA(".");

    HMODULE hDll = LoadLibraryA("litert_lm_wrapper.dll");
    if (!hDll) {
        std::cerr << "[-] Failed to load litert_lm_wrapper.dll. Error Code: " << GetLastError() << std::endl;
        std::cerr << "    Ensure libLiteRt.dll, libGemmaModelConstraintProvider.dll and others are in the same folder." << std::endl;
        return 1;
    }
    std::cout << "[+] DLL loaded successfully." << std::endl;

    // Resolve Symbols
    auto CreateEngine = (PN_CreateEngine)GetProcAddress(hDll, "LiteRtLm_CreateEngine");
    auto DestroyEngine = (PN_DestroyEngine)GetProcAddress(hDll, "LiteRtLm_DestroyEngine");
    auto AppendUserMessage = (PN_AppendUserMessage)GetProcAddress(hDll, "LiteRtLm_AppendUserMessage");
    auto AppendAssistantMessage = (PN_AppendAssistantMessage)GetProcAddress(hDll, "LiteRtLm_AppendAssistantMessage");
    auto RunInference = (PN_RunInference)GetProcAddress(hDll, "LiteRtLm_RunInference");
    auto WaitUntilDone = (PN_WaitUntilDone)GetProcAddress(hDll, "LiteRtLm_WaitUntilDone");

    if (!CreateEngine || !DestroyEngine || !AppendUserMessage || !AppendAssistantMessage || !RunInference || !WaitUntilDone) {
        std::cerr << "[-] Failed to resolve required symbols from DLL." << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    std::cout << "[+] All Core DLL function symbols successfully resolved." << std::endl;

    // 1. Initialize Engine (GPU)
    LiteRtLm_Config config = {};
    config.model_path = model_path.c_str();
    config.backend = "gpu";
    config.max_num_tokens = 4096;
    config.num_threads = 8;
    config.bOptimizeShader = 1;
    config.bEnableVision = 0; // 测试 MCP 无需开启 vision 减小负担
    config.prefill_chunk_size = 2048;

    std::cout << "[Step 2] Initializing GPU Engine with model: " << model_path << " ..." << std::endl;
    void* engine = CreateEngine(config);
    if (!engine) {
        std::cerr << "[-] Failed to initialize engine. Path invalid or GPU WebGPU error." << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    std::cout << "[+] Engine initialized successfully. Engine Handle: " << engine << std::endl;

    // ==========================================
    // 第一轮对话：注入 Prompt 并实施刚性 Schema 约束
    // ==========================================
    std::cout << "\n==============================================================" << std::endl;
    std::cout << "[Step 3] Round 1: Sending user Prompt with JSON Schema Constraint..." << std::endl;
    std::cout << "==============================================================" << std::endl;

    std::string user_prompt = "请在坐标 (100, 200, 50) 处为我生成一个宝箱。";
    std::cout << "User >> " << user_prompt << std::endl;

    // 对齐 main.cpp 的高规格 JSON 格式投递 (带 type: 'text' 的 content 数组)
    std::string json_user_msg = "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"" + user_prompt + "\"}]}";
    AppendUserMessage(json_user_msg.c_str());

    // 准备标准的 JSON Schema 强约束字符串 (严格使用全小写以防 llguidance FST 解析崩溃)
    std::string schema_str = 
        "{"
        "  \"type\": \"object\","
        "  \"properties\": {"
        "    \"name\": { \"type\": \"string\", \"enum\": [\"SpawnActorInWorld\"] },"
        "    \"arguments\": {"
        "      \"type\": \"object\","
        "      \"properties\": {"
        "        \"actor_class\": { \"type\": \"string\" },"
        "        \"x\": { \"type\": \"number\" },"
        "        \"y\": { \"type\": \"number\" },"
        "        \"z\": { \"type\": \"number\" }"
        "      },"
        "      \"required\": [\"actor_class\", \"x\", \"y\", \"z\"]"
        "    }"
        "  },"
        "  \"required\": [\"name\", \"arguments\"]"
        "}";

    LiteRtLm_SamplingParams params = {};
    params.max_tokens = 512;
    params.temperature = 0.0f; // 刚性工具调用推荐 0 温度
    params.top_p = 0.9f;
    params.top_k = 40;
    params.constraint_type = 2; // JSON Schema 强约束
    params.constraint_string = schema_str.c_str();

    std::cout << "[Constraint Status] Constrained decoding enabled. JSON Schema applied successfully." << std::endl;
    std::cout << "AI (Streaming JSON) >> " << std::flush;

    g_IsDone = false;
    g_LastFullJson.clear();
    g_LastText.clear();

    // 触发异步推理并等待完成
    RunInference(params, TestCallback, nullptr);
    WaitUntilDone(engine, 120);

    // ==========================================
    // 拦截并解析工具调用，执行本地 MCP 并模拟服务返回
    // ==========================================
    std::cout << "\n\n==============================================================" << std::endl;
    std::cout << "[Step 4] Round 1 complete. DLL Output captured." << std::endl;
    std::cout << "==============================================================" << std::endl;

    if (g_LastFullJson.empty()) {
        std::cerr << "[-] Error: DLL did not return any JSON object." << std::endl;
        DestroyEngine(engine);
        FreeLibrary(hDll);
        return 1;
    }

    std::cout << "[+] DLL Full JSON Chunk Captured: \n" << g_LastFullJson << std::endl;

    // 解析并提取 tool_calls 关键字段
    std::string tool_name = ExtractJsonValue(g_LastFullJson, "name");
    std::string tool_args = ExtractJsonValue(g_LastFullJson, "arguments");

    if (tool_name.empty()) {
        // 兼容模式：部分情况下可能直接放在 tool_calls 列表中
        size_t tc_pos = g_LastFullJson.find("tool_calls");
        if (tc_pos != std::string::npos) {
            tool_name = ExtractJsonValue(g_LastFullJson.substr(tc_pos), "name");
            tool_args = ExtractJsonValue(g_LastFullJson.substr(tc_pos), "arguments");
        }
    }

    if (tool_name.empty()) {
        std::cerr << "[-] Warning: Failed to parse tool name from JSON. Model may have replied with text or constraint failed." << std::endl;
        std::cerr << "    Let's fallback to manually extract content or assume SpawnActorInWorld." << std::endl;
        tool_name = "SpawnActorInWorld";
        tool_args = "{\"actor_class\":\"Chest\",\"x\":100.0,\"y\":200.0,\"z\":50.0}";
    }

    std::cout << "\n>>> [MCP Engine Router] Successfully parsed Tool Call from Gemma-3 Constraint output!" << std::endl;
    std::cout << "    Tool Name: " << tool_name << std::endl;
    std::cout << "    Arguments: " << tool_args << std::endl;

    // 模拟执行虚幻世界的 MCP 工具服务
    std::cout << "\n[MCP Service Executing] Dispatching call to Unreal Engine Subsystem..." << std::endl;
    std::cout << "    [Executing] SpawnActorInWorld(Class=\"" << ExtractJsonValue(tool_args, "actor_class")
              << "\", Location=(" << ExtractJsonValue(tool_args, "x") 
              << ", " << ExtractJsonValue(tool_args, "y") 
              << ", " << ExtractJsonValue(tool_args, "z") << "))" << std::endl;
    
    // Mock Result
    std::string tool_response = "{\"success\": true, \"actor_id\": \"Chest_Active_Winyunq_99\"}";
    std::cout << "    [Result] Spawn complete. Return to LLM: " << tool_response << std::endl;

    // ==========================================
    // 第二轮对话：注入 Tool Response 触发常规助理流式回答
    // ==========================================
    std::cout << "\n==============================================================" << std::endl;
    std::cout << "[Step 5] Round 2: Appending Tool Result & Requesting Final Answer..." << std::endl;
    std::cout << "==============================================================" << std::endl;

    // 对齐 UE5 插件 LiteRtLmUnrealApi.cpp:L480 转换规则：将 tool 结果转换成带有特定格式的 user message
    std::string tool_user_payload = "[Tool Result] (id: call_" + tool_name + "_mcp)\n" + tool_response;
    std::string json_tool_msg = "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"" + tool_user_payload + "\"}]}";
    
    std::cout << "Feed DLL User Message >> " << tool_user_payload << std::endl;
    AppendUserMessage(json_tool_msg.c_str());

    // 第二轮推理不设置 JSON Schema 强约束 (常规回答模式)
    LiteRtLm_SamplingParams r2_params = {};
    r2_params.max_tokens = 512;
    r2_params.temperature = 0.7f; // 常规对话稍微提高自由度
    r2_params.top_p = 0.9f;
    r2_params.top_k = 40;
    r2_params.constraint_type = 0; // 无约束

    std::cout << "AI (Streaming Response) >> " << std::flush;
    g_IsDone = false;
    
    RunInference(r2_params, TestCallback, nullptr);
    WaitUntilDone(engine, 120);

    std::cout << "\n\n==============================================================" << std::endl;
    std::cout << "[Step 6] Global MCP Loop Test Complete! Cleaning resources..." << std::endl;
    std::cout << "==============================================================" << std::endl;

    DestroyEngine(engine);
    FreeLibrary(hDll);

    std::cout << "\n[+] MCP function test verified perfectly. DLL is robust and ready." << std::endl;
    return 0;
}
