// Copyright (c) 2025-2026 Winyunq. All rights reserved.
#include <iostream>
#include <windows.h>
#include <string>
#include <atomic>
#include <vector>
#include <filesystem>
#include <fstream>
#include <gdiplus.h>

#pragma comment(lib, "gdiplus.lib")

// --- Wrapper Types ---
typedef struct {
    const char* model_path;
    const char* backend;
    int max_num_tokens;
    int num_threads;
    int bEnableBenchmark;
    int bOptimizeShader;
    int bEnableVision;
    int bEnableAudio;
    int prefill_chunk_size;
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
typedef void (*PN_AppendUserMessage)(const char* json_msg);
typedef void (*PN_AppendAssistantMessage)(const char* text);
typedef void (*PN_RunInference)(LiteRtLm_SamplingParams params, LiteRtLmCallback callback, void* user_ptr);
typedef void (*PN_StopMessage)();
typedef int (*PN_WaitUntilDone)(void* engine_ptr, int timeout_sec);

// 新增物理 KV 缓存接口
typedef int (*PN_GetKVCache)(void* data_ptr, size_t* out_size);
typedef int (*PN_SetKVCache)(const void* data_ptr, size_t size);

// --- Helpers ---

std::string EscapePath(const std::string& path) {
    std::string out;
    for (char c : path) {
        if (c == '\\') out += "\\\\";
        else out += c;
    }
    return out;
}

std::string EscapeJsonString(const std::string& input) {
    std::string out;
    for (char c : input) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

int GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
    UINT num = 0;
    UINT size = 0;
    Gdiplus::GetImageEncodersSize(&num, &size);
    if (size == 0) return -1;
    Gdiplus::ImageCodecInfo* pImageCodecInfo = (Gdiplus::ImageCodecInfo*)(malloc(size));
    if (pImageCodecInfo == NULL) return -1;
    Gdiplus::GetImageEncoders(num, size, pImageCodecInfo);
    for (UINT j = 0; j < num; ++j) {
        if (wcscmp(pImageCodecInfo[j].MimeType, format) == 0) {
            *pClsid = pImageCodecInfo[j].Clsid;
            free(pImageCodecInfo);
            return j;
        }
    }
    free(pImageCodecInfo);
    return -1;
}

std::string Trim(const std::string& str) {
    if (str.empty()) return "";
    size_t first = str.find_first_not_of(" \t\r\n\"");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n\"");
    return str.substr(first, (last - first + 1));
}

std::wstring AnsiToWstring(const std::string& str) {
    if (str.empty()) return L"";
    std::string trimmed = Trim(str);
    int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, trimmed.c_str(), -1, NULL, 0);
    if (len > 0) {
        std::wstring wstrTo(len - 1, 0);
        MultiByteToWideChar(CP_UTF8, 0, trimmed.c_str(), -1, &wstrTo[0], len);
        return wstrTo;
    }
    len = MultiByteToWideChar(CP_ACP, 0, trimmed.c_str(), -1, NULL, 0);
    if (len > 0) {
        std::wstring wstrTo(len - 1, 0);
        MultiByteToWideChar(CP_ACP, 0, trimmed.c_str(), -1, &wstrTo[0], len);
        return wstrTo;
    }
    std::wstring wstrTo(trimmed.begin(), trimmed.end());
    return wstrTo;
}

std::string WstringToAnsi(const std::wstring& wstr) {
    if (wstr.empty()) return "";
    int len = WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, NULL, 0, NULL, NULL);
    if (len <= 0) return "";
    std::string strTo(len - 1, 0);
    WideCharToMultiByte(CP_ACP, 0, wstr.c_str(), -1, &strTo[0], len, NULL, NULL);
    return strTo;
}

std::string GetAppTempImagePath() {
    wchar_t tempPath[MAX_PATH];
    if (GetTempPathW(MAX_PATH, tempPath) == 0) {
        return "winyunq_temp_multimodal.png";
    }
    std::wstring wOut = std::wstring(tempPath) + L"winyunq_temp_multimodal.png";
    return WstringToAnsi(wOut);
}

std::string ResizeImageGDI(const std::string& inputPath, int maxDim = 512) {
    std::string trimmedInput = Trim(inputPath);
    std::string outputPath = GetAppTempImagePath();
    bool success = false;
    {
        std::wstring wInputPath = AnsiToWstring(trimmedInput);
        std::wstring wOutputPath = AnsiToWstring(outputPath);

        Gdiplus::Bitmap* source = Gdiplus::Bitmap::FromFile(wInputPath.c_str());
        if (source && source->GetLastStatus() == Gdiplus::Ok) {
            int width = source->GetWidth();
            int height = source->GetHeight();

            if (width > maxDim || height > maxDim) {
                float scale = (float)maxDim / (width > height ? width : height);
                int newWidth = (int)(width * scale);
                int newHeight = (int)(height * scale);

                Gdiplus::Bitmap* target = new Gdiplus::Bitmap(newWidth, newHeight, PixelFormat32bppARGB);
                Gdiplus::Graphics* g = Gdiplus::Graphics::FromImage(target);
                g->SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBilinear);
                g->DrawImage(source, 0, 0, newWidth, newHeight);

                CLSID pngClsid;
                GetEncoderClsid(L"image/png", &pngClsid);
                target->Save(wOutputPath.c_str(), &pngClsid, NULL);

                delete g;
                delete target;
                std::cout << "[Image Resize] Bilinear downsampled " << width << "x" << height 
                          << " -> " << newWidth << "x" << newHeight << " to bypass VRAM OOM!" << std::endl;
                success = true;
            } else {
                CLSID pngClsid;
                GetEncoderClsid(L"image/png", &pngClsid);
                source->Save(wOutputPath.c_str(), &pngClsid, NULL);
                success = true;
            }
            delete source;
        } else {
            if (source) delete source;
        }
    }

    if (success) {
        int retries = 0;
        while (retries < 50) {
            try {
                if (std::filesystem::exists(outputPath) && std::filesystem::file_size(outputPath) > 100) {
                    std::ifstream file(outputPath, std::ios::binary);
                    if (file.is_open()) {
                        file.close();
                        break; 
                    }
                }
            } catch (...) {}
            Sleep(10);
            retries++;
        }
    }
    return success ? outputPath : inputPath;
}

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

void RunKVCachePhysicalTest(void* engine, 
                            PN_GetKVCache GetKVCache, 
                            PN_SetKVCache SetKVCache, 
                            PN_AppendUserMessage AppendUserMessage, 
                            PN_RunInference RunInference, 
                            PN_WaitUntilDone WaitUntilDone) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "[KV Cache Test] Phase 1: Injection Memory..." << std::endl;
    std::cout << "==============================================" << std::endl;

    std::string secret_msg = "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Please remember this secret number: 5201314. Do not forget it!\"}]}";
    AppendUserMessage(secret_msg.c_str());

    LiteRtLm_SamplingParams test_params = {};
    test_params.max_tokens = 64;
    test_params.temperature = 0.0f; // 贪婪采样确保一致性

    std::cout << "AI >> " << std::flush;
    g_IsDone = false;
    RunInference(test_params, MyCallback, nullptr);
    WaitUntilDone(engine, 600);

    // 开始物理导出备份
    std::cout << "\n\n[KV Cache Test] Phase 2: Backing up physical KV Cache from VRAM..." << std::endl;
    size_t kv_size = 0;
    int r1 = GetKVCache(nullptr, &kv_size);
    if (r1 != 0 || kv_size == 0) {
        std::cerr << "Failed to query physical KV cache size, code: " << r1 << std::endl;
        return;
    }

    std::vector<char> kv_backup(kv_size);
    int r2 = GetKVCache(kv_backup.data(), &kv_size);
    if (r2 != 0) {
        std::cerr << "Failed to export physical KV cache, code: " << r2 << std::endl;
        return;
    }
    std::cout << "[KV Cache Test] Successfully backed up " << kv_size << " bytes of physical attention key-value tensors." << std::endl;

    // 强行清空/擦除 KV 缓存
    std::cout << "\n[KV Cache Test] Phase 3: Erasing current active KV cache..." << std::endl;
    int r3 = SetKVCache(nullptr, 0);
    if (r3 != 0) {
        std::cerr << "Failed to clear active KV cache, code: " << r3 << std::endl;
    } else {
        std::cout << "[KV Cache Test] Active KV cache has been completely cleared (Memory Reset)." << std::endl;
    }

    // 提问验证模型是否忘记
    std::cout << "\n[KV Cache Test] Phase 4: Verification without KV Cache..." << std::endl;
    std::string ask_msg1 = "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"What was the secret number I just asked you to remember?\"}]}";
    AppendUserMessage(ask_msg1.c_str());

    std::cout << "AI >> " << std::flush;
    g_IsDone = false;
    RunInference(test_params, MyCallback, nullptr);
    WaitUntilDone(engine, 600);

    // 物理还原先前的显存大包
    std::cout << "\n\n[KV Cache Test] Phase 5: Restoring physical KV Cache to VRAM..." << std::endl;
    int r4 = SetKVCache(kv_backup.data(), kv_backup.size());
    if (r4 != 0) {
        std::cerr << "Failed to restore physical KV cache, code: " << r4 << std::endl;
    } else {
        std::cout << "[KV Cache Test] Successfully restored " << kv_backup.size() << " bytes of physical attention state into GPU memory!" << std::endl;
    }

    // 再次提问验证模型是否瞬间回忆起来
    std::cout << "\n[KV Cache Test] Phase 6: Verification WITH Restored KV Cache..." << std::endl;
    std::string ask_msg2 = "{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"What was the secret number I just asked you to remember? Answer it directly.\"}]}";
    AppendUserMessage(ask_msg2.c_str());

    std::cout << "AI >> " << std::flush;
    g_IsDone = false;
    RunInference(test_params, MyCallback, nullptr);
    WaitUntilDone(engine, 600);
    std::cout << "\n==============================================" << std::endl;
    std::cout << "[KV Cache Test] Verification complete." << std::endl;
    std::cout << "==============================================\n" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== WinyunqDebug: Global Single-Session & Direct VRAM Lock KV Cache Testing ===" << std::endl;

    bool bTestKVCache = false;
    if (argc > 1 && std::string(argv[1]) == "--test-kv") {
        bTestKVCache = true;
    }

    // 全局高能初始化 GDI+ 资源
    ULONG_PTR gdiplusToken;
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    HMODULE hDll = LoadLibraryA("litert_lm_wrapper.dll");
    if (!hDll) {
        std::cerr << "Failed to load DLL. Error: " << GetLastError() << std::endl;
        Gdiplus::GdiplusShutdown(gdiplusToken);
        return 1;
    }

    char dllPath[MAX_PATH];
    if (GetModuleFileNameA(hDll, dllPath, MAX_PATH)) {
        std::cout << "[Debug] Loaded DLL Absolute Path: " << dllPath << std::endl;
    }

    auto CreateEngine = (PN_CreateEngine)GetProcAddress(hDll, "LiteRtLm_CreateEngine");
    auto DestroyEngine = (PN_DestroyEngine)GetProcAddress(hDll, "LiteRtLm_DestroyEngine");
    auto AppendUserMessage = (PN_AppendUserMessage)GetProcAddress(hDll, "LiteRtLm_AppendUserMessage");
    auto RunInference = (PN_RunInference)GetProcAddress(hDll, "LiteRtLm_RunInference");
    auto WaitUntilDone = (PN_WaitUntilDone)GetProcAddress(hDll, "LiteRtLm_WaitUntilDone");
    
    // 动态获取 KV Cache 函数指针
    auto GetKVCache = (PN_GetKVCache)GetProcAddress(hDll, "LiteRtLm_GetKVCache");
    auto SetKVCache = (PN_SetKVCache)GetProcAddress(hDll, "LiteRtLm_SetKVCache");

    if (!CreateEngine || !RunInference || !WaitUntilDone || !AppendUserMessage || !GetKVCache || !SetKVCache) {
        std::cerr << "Failed to resolve symbols in wrapper DLL." << std::endl;
        if (!CreateEngine) std::cerr << "-> LiteRtLm_CreateEngine not found!" << std::endl;
        if (!RunInference) std::cerr << "-> LiteRtLm_RunInference not found!" << std::endl;
        if (!WaitUntilDone) std::cerr << "-> LiteRtLm_WaitUntilDone not found!" << std::endl;
        if (!AppendUserMessage) std::cerr << "-> LiteRtLm_AppendUserMessage not found!" << std::endl;
        if (!GetKVCache) std::cerr << "-> LiteRtLm_GetKVCache not found!" << std::endl;
        if (!SetKVCache) std::cerr << "-> LiteRtLm_SetKVCache not found!" << std::endl;
        FreeLibrary(hDll);
        Gdiplus::GdiplusShutdown(gdiplusToken);
        return 1;
    }

    // 1. Initialize Engine (GPU)
    LiteRtLm_Config config = {};
    config.model_path = "D:\\gemma-4-E4B-it.litertlm";
    config.backend = "gpu";
    config.max_num_tokens = 65536;
    config.bOptimizeShader = 1;
    /// 启用多模态视觉（Vision）引擎以支持图像推理，防止拖入图片时发生空指针崩溃
    config.bEnableVision = 1;
    config.prefill_chunk_size = 32768; // 单次处理的最长上下文，作为初始化参数传入！

    std::cout << "Initializing Engine (GPU) & Global Single Conversation..." << std::endl;
    void* engine = CreateEngine(config);
    if (!engine) {
        std::cerr << "Failed to create engine." << std::endl;
        FreeLibrary(hDll);
        Gdiplus::GdiplusShutdown(gdiplusToken);
        return 1;
    }

    // 2. 根据命令行参数决定是执行物理 KV 缓存回忆测试，还是直接开启 Chat Loop
    if (bTestKVCache) {
        RunKVCachePhysicalTest(engine, GetKVCache, SetKVCache, AppendUserMessage, RunInference, WaitUntilDone);
    } else {
        // 3. Chat Loop
        std::string input;
        std::cout << "Starting normal Chat Loop. Type 'exit' to exit." << std::endl;
        std::cout << "Hint: Drag an image or type 'Path, Prompt' to test multi-modal." << std::endl;
        while (true) {
            std::cout << "\nUser >> ";
            if (!std::getline(std::cin, input) || input == "exit") break;
            if (input.empty()) continue;

            bool hasImage = false;
            std::string path, prompt;

            const char* exts[] = {".png", ".jpg", ".jpeg", ".bmp"};
            for (const char* ext : exts) {
                size_t pos = input.find(ext);
                if (pos != std::string::npos) {
                    size_t endPos = pos + strlen(ext);
                    path = input.substr(0, endPos);
                    if (path.front() == '"') path = path.substr(1);
                    if (path.back() == '"') path.pop_back();

                    if (endPos < input.size()) {
                        prompt = input.substr(endPos);
                        while (!prompt.empty()) {
                            if (prompt[0] == ' ' || prompt[0] == ',' || prompt[0] == ';' || prompt[0] == ':') {
                                prompt = prompt.substr(1);
                            } else if (prompt.size() >= 3 && prompt.substr(0, 3) == "，") {
                                prompt = prompt.substr(3); 
                            } else {
                                break;
                            }
                        }
                    }
                    if (prompt.empty()) prompt = "Please describe this image.";
                    
                    std::string cleanedPath = Trim(path);
                    if (std::filesystem::exists(cleanedPath)) {
                        hasImage = true;
                        path = cleanedPath;
                    } else {
                        std::cout << "[Warning: Image file not found at \"" << cleanedPath << "\", falling back to pure text mode]" << std::endl;
                        hasImage = false;
                    }
                    break;
                }
            }

            if (hasImage) {
                std::string resizedPath = ResizeImageGDI(path, 256);
                std::string escapedPath = EscapePath(resizedPath);
                std::cout << "[Image Found: " << path << "]" << std::endl;
                
                std::string json_msg = "{\"role\": \"user\", \"content\": ["
                                       "{\"type\": \"image\", \"path\": \"" + escapedPath + "\"},"
                                       "{\"type\": \"text\", \"text\": \"" + prompt + "\"}"
                                       "]}";
                
                AppendUserMessage(json_msg.c_str());
            } else {
                std::string json_msg = "{\"role\": \"user\", \"content\": ["
                                       "{\"type\": \"text\", \"text\": \"" + EscapeJsonString(input) + "\"}"
                                       "]}";
                AppendUserMessage(json_msg.c_str());
            }

            LiteRtLm_SamplingParams params = {};
            params.max_tokens = 512;
            params.temperature = 0.7f;

            std::cout << "AI >> " << std::flush;
            g_IsDone = false;
            RunInference(params, MyCallback, nullptr);

            WaitUntilDone(engine, 600);
        }
    }

    DestroyEngine(engine);
    FreeLibrary(hDll);
    
    Gdiplus::GdiplusShutdown(gdiplusToken);
    return 0;
}
