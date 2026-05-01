#include <iostream>
#include <windows.h>
#include <string>
#include <atomic>

// --- Wrapper Types ---
typedef struct {
    const char* model_path;
    const char* backend;
    int max_num_tokens;
    int num_threads;
    int bEnableBenchmark;
    int bOptimizeShader;
} LiteRtLm_Config;

typedef struct {
    float temperature;
    float top_p;
    int top_k;
    int max_tokens;
    int constraint_type;
    const char* constraint_string;
} LiteRtLm_SamplingParams;

typedef struct {
    const char* text_chunk;
    const char* full_json_chunk;
    const char* error_msg;
    int bIsDone;
    float tokens_per_sec;
} LiteRtLm_Result;

typedef void (*LiteRtLmCallback)(LiteRtLm_Result result, void* user_ptr);

// --- Function Pointers ---
typedef const char* (*PN_GetAvailableBackends)();
typedef void* (*PN_CreateEngine)(LiteRtLm_Config config);
typedef void (*PN_DestroyEngine)(void* engine_ptr);
typedef void* (*PN_CreateConversation)(void* engine_ptr);
typedef void (*PN_DestroyConversation)(void* conv_ptr);
typedef void (*PN_AppendUserMessage)(void* conv_ptr, const char* text);
typedef void (*PN_RunInference)(void* conv_ptr, LiteRtLm_SamplingParams params, LiteRtLmCallback callback, void* user_ptr);
typedef int (*PN_WaitUntilDone)(void* engine_ptr, int timeout_sec);

// --- Global State for Callback ---
std::atomic<bool> g_IsDone{false};

void MyCallback(LiteRtLm_Result result, void* user_ptr) {
    if (result.text_chunk) {
        std::cout << result.text_chunk << std::flush;
    }
    if (result.error_msg) {
        std::cerr << "\n[ERROR] " << result.error_msg << std::endl;
    }
    if (result.bIsDone) {
        g_IsDone = true;
        if (result.tokens_per_sec > 0) {
            std::cout << "\n[Speed: " << result.tokens_per_sec << " tokens/sec]" << std::endl;
        }
    }
}

int main() {
    std::cout << "=== WinyunqDebug: Full GPU Inference Test ===" << std::endl;

    HMODULE hDll = LoadLibraryA("litert_lm_wrapper.dll");
    if (!hDll) {
        std::cerr << "Failed to load DLL. Error: " << GetLastError() << std::endl;
        return 1;
    }

    // Load Functions
    auto CreateEngine = (PN_CreateEngine)GetProcAddress(hDll, "LiteRtLm_CreateEngine");
    auto DestroyEngine = (PN_DestroyEngine)GetProcAddress(hDll, "LiteRtLm_DestroyEngine");
    auto CreateConversation = (PN_CreateConversation)GetProcAddress(hDll, "LiteRtLm_CreateConversation");
    auto DestroyConversation = (PN_DestroyConversation)GetProcAddress(hDll, "LiteRtLm_DestroyConversation");
    auto AppendUserMessage = (PN_AppendUserMessage)GetProcAddress(hDll, "LiteRtLm_AppendUserMessage");
    auto RunInference = (PN_RunInference)GetProcAddress(hDll, "LiteRtLm_RunInference");
    auto WaitUntilDone = (PN_WaitUntilDone)GetProcAddress(hDll, "LiteRtLm_WaitUntilDone");

    if (!CreateEngine || !RunInference || !WaitUntilDone) {
        std::cerr << "Failed to resolve symbols." << std::endl;
        return 1;
    }

    // 1. Initialize Engine
    LiteRtLm_Config config = {};
    config.model_path = "D:\\gemma-4-E4B-it.litertlm";
    config.backend = "gpu"; // 强制 GPU
    config.max_num_tokens = 2048;
    config.bOptimizeShader = 1;

    std::cout << "Initializing Engine (GPU)..." << std::endl;
    void* engine = CreateEngine(config);
    if (!engine) {
        std::cerr << "Failed to create engine. Check model path or GPU support." << std::endl;
        return 1;
    }

    // 2. Create Conversation
    void* conv = CreateConversation(engine);

    // 3. Chat Loop
    std::string input;
    while (true) {
        std::cout << "\nUser >> ";
        if (!std::getline(std::cin, input) || input == "exit") break;
        if (input.empty()) continue;

        AppendUserMessage(conv, input.c_str());

        LiteRtLm_SamplingParams params = {};
        params.max_tokens = 512;
        params.temperature = 0.7f;
        params.top_p = 0.9f;
        params.top_k = 40;

        std::cout << "AI >> ";
        g_IsDone = false;
        RunInference(conv, params, MyCallback, nullptr);

        // Wait for completion
        int waitRes = WaitUntilDone(engine, 600); 
        if (waitRes != 0) {
            std::cerr << "WaitUntilDone failed or timed out: " << waitRes << std::endl;
        }
    }

    // Cleanup
    DestroyConversation(conv);
    DestroyEngine(engine);
    FreeLibrary(hDll);

    std::cout << "\nVerification finished." << std::endl;
    return 0;
}
