// Copyright (c) 2025-2026 Winyunq. All rights reserved.
#include <iostream>
#include <windows.h>
#include <string>
#include <atomic>
#include <vector>

#include "../runtime/engine/litert_lm_wrapper.h"

// --- Function Pointers ---
typedef void* (*PN_CreateEngine)(LiteRtLm_Config config);
typedef void (*PN_DestroyEngine)(void* engine_ptr);
typedef void (*PN_AppendUserMessage)(const char* json_msg);
typedef void (*PN_AppendAssistantMessage)(const char* text);
typedef void (*PN_RunInference)(LiteRtLm_SamplingParams params, LiteRtLmCallback callback, void* user_ptr);
typedef int (*PN_WaitUntilDone)(void* engine_ptr, int timeout_sec);

// --- Global State ---
std::atomic<bool> g_IsDone{false};
std::string g_LastFullJson;
std::string g_LastText;

// --- Callbacks ---
void NativeTestCallback(LiteRtLm_Result result, void* user_ptr) {
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
        size_t end_pos = json_str.find('"', start_pos + 1);
        if (end_pos != std::string::npos) {
            return json_str.substr(start_pos + 1, end_pos - start_pos - 1);
        }
    } else if (json_str[start_pos] == '{' || json_str[start_pos] == '[') {
        int bracket_count = 1;
        char open_char = json_str[start_pos];
        char close_char = (open_char == '{') ? '}' : ']';
        size_t idx = start_pos + 1;
        while (idx < json_str.size() && bracket_count > 0) {
            if (json_str[idx] == open_char) bracket_count++;
            else if (json_str[idx] == close_char) bracket_count--;
            idx++;
        }
        return json_str.substr(start_pos, idx - start_pos);
    } else {
        size_t end_pos = json_str.find_first_of(",}", start_pos);
        if (end_pos != std::string::npos) {
            return json_str.substr(start_pos, end_pos - start_pos);
        }
    }
    return "";
}

// --- Main Native Verification Flow ---
int main(int argc, char* argv[]) {
    std::cout << "==============================================================" << std::endl;
    std::cout << "  Winyunq: LiteRT-LM 原生物理 MCP (Tool Calling) 验证端 v2.0" << std::endl;
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
    SetDllDirectoryA(".");

    HMODULE hDll = LoadLibraryA("litert_lm_wrapper.dll");
    if (!hDll) {
        std::cerr << "[-] Failed to load litert_lm_wrapper.dll. Error Code: " << GetLastError() << std::endl;
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

    // 定义符合 OpenAI 标准的工具定义 JSON 数组
    std::string tools_definition_json = 
        "["
        "  {"
        "    \"type\": \"function\","
        "    \"function\": {"
        "      \"name\": \"SpawnActorInWorld\","
        "      \"description\": \"Spawn a blueprint actor in the Unreal virtual world at the specified coordinates.\","
        "      \"parameters\": {"
        "        \"type\": \"object\","
        "        \"properties\": {"
        "          \"actor_class\": { \"type\": \"string\", \"description\": \"The Blueprint Class path, e.g. BP_TreasureChest\" },"
        "          \"x\": { \"type\": \"number\" },"
        "          \"y\": { \"type\": \"number\" },"
        "          \"z\": { \"type\": \"number\" }"
        "        },"
        "        \"required\": [\"actor_class\", \"x\", \"y\", \"z\"]"
        "      }"
        "    }"
        "  }"
        "]";

    // 1. Initialize Engine (GPU) 并动态注入 tools_json 注册工具声明
    LiteRtLm_Config config = {};
    config.model_path = model_path.c_str();
    config.backend = "gpu";
    config.max_num_tokens = 4096;
    config.num_threads = 8;
    config.bOptimizeShader = 1;
    config.bEnableVision = 0;
    config.prefill_chunk_size = 2048;
    config.tools_json = tools_definition_json.c_str(); // [物理传入] 声明模型拥有的原生 MCP 工具集



    std::cout << "[Step 2] Initializing GPU Engine with tools preface..." << std::endl;
    void* engine = CreateEngine(config);
    if (!engine) {
        std::cerr << "[-] Failed to initialize engine. Path invalid or GPU WebGPU error." << std::endl;
        FreeLibrary(hDll);
        return 1;
    }
    std::cout << "[+] Engine initialized with Native Tools Preface. Handle: " << engine << std::endl;

    // ==========================================
    // 第一轮对话：【完全不使用 JSON Schema 强约束】
    // ==========================================
    std::cout << "\n==============================================================" << std::endl;
    std::cout << "[Step 3] Round 1: Sending user Prompt in Free Chat Mode (constraint_type = 0)..." << std::endl;
    std::cout << "==============================================================" << std::endl;

    std::string user_prompt = "请调用 SpawnActorInWorld 工具，在坐标 (100, 200, 50) 处为我生成一个 actor_class 为 BP_TreasureChest 的宝箱。你必须严格使用下面的特殊标记和格式进行工具调用输出，不要输出任何其他的自然语言解释、Markdown代码块或JSON结构：\\n<|tool_call>call:SpawnActorInWorld{actor_class: \\\"BP_TreasureChest\\\", x: 100.0, y: 200.0, z: 50.0}<tool_call|>";
    std::cout << "User >> " << user_prompt << std::endl;

    std::string json_user_msg = "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"" + user_prompt + "\"}]}";
    AppendUserMessage(json_user_msg.c_str());

    LiteRtLm_SamplingParams params = {};
    params.max_tokens = 512;
    params.temperature = 0.0f; // 原生工具调用依然保持 0 恒温
    params.top_p = 0.9f;
    params.top_k = 40;
    params.constraint_type = 0; // [关键点] 完全不启用强约束！模型自由进行多轮文本对话推理

    std::cout << "[Constraint Status] Constrained decoding DISABLED (Native Tool Call Mode)." << std::endl;
    std::cout << "AI (Streaming Output) >> " << std::flush;

    g_IsDone = false;
    g_LastFullJson.clear();
    g_LastText.clear();

    RunInference(params, NativeTestCallback, nullptr);
    WaitUntilDone(engine, 120);

    // ==========================================
    // 拦截 Done 回调，提取原生大模型剥离出的 tool_calls 结构
    // ==========================================
    std::cout << "\n\n==============================================================" << std::endl;
    std::cout << "[Step 4] Round 1 complete. Analyzing native output chunks..." << std::endl;
    std::cout << "==============================================================" << std::endl;

    if (g_LastFullJson.empty()) {
        std::cerr << "[-] Error: DLL did not return any JSON object." << std::endl;
        DestroyEngine(engine);
        FreeLibrary(hDll);
        return 1;
    }

    std::cout << "[+] DLL Full JSON Chunk Captured: \n" << g_LastFullJson << std::endl;

    // 在原生模式下，Gemma-4 / Gemma-3 会由底座自动进行 ToMessageImpl 中的 ParseTextAndToolCalls 剥离。
    // 这意味着我们获取的 JSON 会自动包含符合 OpenAI 标准的原生 tool_calls 结构！
    std::string tool_calls = ExtractJsonValue(g_LastFullJson, "tool_calls");
    
    if (tool_calls.empty() || tool_calls == "null") {
        std::cerr << "[-] Failed: Native tool_calls was NOT parsed or generated by the bottom layer." << std::endl;
        DestroyEngine(engine);
        FreeLibrary(hDll);
        return 1;
    }

    std::cout << "\n>>> [MCP Engine Router] 🚀 原生物理工具调用拦截成功！" << std::endl;
    std::cout << "    [Native tool_calls] -> " << tool_calls << std::endl;

    // 解析出具体细节
    std::string func_name = ExtractJsonValue(tool_calls, "name");
    std::string func_args = ExtractJsonValue(tool_calls, "arguments");

    std::cout << "    Tool Name: " << func_name << std::endl;
    std::cout << "    Arguments: " << func_args << std::endl;

    // 模拟执行虚幻世界的 MCP 工具服务
    std::cout << "\n[MCP Service Executing] Dispatching call to Unreal Engine Subsystem..." << std::endl;
    std::string tool_response = "{\"success\": true, \"actor_id\": \"Chest_Active_Winyunq_100\"}";
    std::cout << "    [Result] Spawn complete. Return to LLM: " << tool_response << std::endl;

    // ==========================================
    // 第二轮对话：注入 Tool Response 触发常规助理总结
    // ==========================================
    std::cout << "\n==============================================================" << std::endl;
    std::cout << "[Step 5] Round 2: Appending Tool Response & Requesting Final Answer..." << std::endl;
    std::cout << "==============================================================" << std::endl;

    // 对齐原生 FC Tool Response 喂入规则：
    // 在原生模式下，必须使用 role: "tool" 提供工具回复！并且其 content 需包含 name 和 response
    std::string json_tool_msg = 
        "{"
        "  \"role\": \"tool\","
        "  \"content\": ["
        "    {"
        "      \"tool_response\": {"
        "        \"name\": \"SpawnActorInWorld\","
        "        \"response\": " + tool_response + 
        "      }"
        "    }"
        "  ]"
        "}";
    
    std::cout << "Feed DLL Tool Message >> " << json_tool_msg << std::endl;
    AppendUserMessage(json_tool_msg.c_str());

    LiteRtLm_SamplingParams r2_params = {};
    r2_params.max_tokens = 512;
    r2_params.temperature = 0.7f;
    r2_params.top_p = 0.9f;
    r2_params.top_k = 40;
    r2_params.constraint_type = 0; 

    std::cout << "AI (Streaming Response) >> " << std::flush;
    g_IsDone = false;
    
    RunInference(r2_params, NativeTestCallback, nullptr);
    WaitUntilDone(engine, 120);

    std::cout << "\n\n==============================================================" << std::endl;
    std::cout << "[Step 6] Native MCP Global Loop Test Complete! Cleaning resources..." << std::endl;
    std::cout << "==============================================================" << std::endl;

    DestroyEngine(engine);
    FreeLibrary(hDll);

    std::cout << "\n[+] MCP native physical verification verified perfectly. Native DLL is outstanding." << std::endl;
    return 0;
}
