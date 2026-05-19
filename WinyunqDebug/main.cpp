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
typedef void (*PN_AppendUserMessage)(void* conv_ptr, const char* json_msg);
typedef void (*PN_RunInference)(void* conv_ptr, LiteRtLm_SamplingParams params, LiteRtLmCallback callback, void* user_ptr);
typedef int (*PN_WaitUntilDone)(void* engine_ptr, int timeout_sec);

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
    // 1. 优先尝试以 UTF-8 编码进行高保真转换 (探测现代 Windows PowerShell/CMD 的输入流)
    int len = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, trimmed.c_str(), -1, NULL, 0);
    if (len > 0) {
        std::wstring wstrTo(len - 1, 0);
        MultiByteToWideChar(CP_UTF8, 0, trimmed.c_str(), -1, &wstrTo[0], len);
        return wstrTo;
    }
    // 2. 失败则退回到 CP_ACP (本地 ANSI 代码页，如中文 GBK)
    len = MultiByteToWideChar(CP_ACP, 0, trimmed.c_str(), -1, NULL, 0);
    if (len > 0) {
        std::wstring wstrTo(len - 1, 0);
        MultiByteToWideChar(CP_ACP, 0, trimmed.c_str(), -1, &wstrTo[0], len);
        return wstrTo;
    }
    // 3. Fallback 退让逻辑，避免极端截断
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
        // 自动探测并高保真还原宽字符路径
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
                        break; // 100% OK
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

int main() {
    std::cout << "=== WinyunqDebug: Multi-modal (Path Escaping & GDI Bilinear Downsampling) ===" << std::endl;

    // 全局高能初始化 GDI+ 资源 (规避反复 Startup / Shutdown 带来的高额 CPU 延迟)
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
    auto CreateConversation = (PN_CreateConversation)GetProcAddress(hDll, "LiteRtLm_CreateConversation");
    auto DestroyConversation = (PN_DestroyConversation)GetProcAddress(hDll, "LiteRtLm_DestroyConversation");
    auto AppendUserMessage = (PN_AppendUserMessage)GetProcAddress(hDll, "LiteRtLm_AppendUserMessage");
    auto RunInference = (PN_RunInference)GetProcAddress(hDll, "LiteRtLm_RunInference");
    auto WaitUntilDone = (PN_WaitUntilDone)GetProcAddress(hDll, "LiteRtLm_WaitUntilDone");

    if (!CreateEngine || !RunInference || !WaitUntilDone || !AppendUserMessage) {
        std::cerr << "Failed to resolve symbols." << std::endl;
        return 1;
    }

    // 1. Initialize Engine
    LiteRtLm_Config config = {};
    config.model_path = "D:\\gemma-4-E4B-it.litertlm";
    config.backend = "gpu";
    config.max_num_tokens = 65536;
    config.bOptimizeShader = 1;

    std::cout << "Initializing Engine (GPU)..." << std::endl;
    void* engine = CreateEngine(config);
    if (!engine) {
        std::cerr << "Failed to create engine." << std::endl;
        return 1;
    }

    void* conv = CreateConversation(engine);

    // 3. Chat Loop
    std::string input;
    std::cout << "\nHint: Drag an image or type 'Path, Prompt' to test multi-modal." << std::endl;
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
                    // 仅跳过空格、英文逗号、分号，绝不跳过中文字符
                    while (!prompt.empty()) {
                        if (prompt[0] == ' ' || prompt[0] == ',' || prompt[0] == ';' || prompt[0] == ':') {
                            prompt = prompt.substr(1);
                        } else if (prompt.size() >= 3 && prompt.substr(0, 3) == "，") {
                            prompt = prompt.substr(3); // 过滤中文逗号
                        } else {
                            break;
                        }
                    }
                }
                if (prompt.empty()) prompt = "Please describe this image.";
                
                // 物理校验文件路径是否存在，提供顶级的鲁棒性
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
            // 用 Windows GDI+ 双线性缩放高分辨率大图片，规避 VRAM/Prefill 极端 OOM 闪退！
            std::string resizedPath = ResizeImageGDI(path, 256);
            std::string escapedPath = EscapePath(resizedPath);
            std::cout << "[Image Found: " << path << "]" << std::endl;
            
            std::string json_msg = "{\"role\": \"user\", \"content\": ["
                                   "{\"type\": \"image\", \"path\": \"" + escapedPath + "\"},"
                                   "{\"type\": \"text\", \"text\": \"" + prompt + "\"}"
                                   "]}";
            
            AppendUserMessage(conv, json_msg.c_str());
        } else {
            std::string json_msg = "{\"role\": \"user\", \"content\": ["
                                   "{\"type\": \"text\", \"text\": \"" + EscapeJsonString(input) + "\"}"
                                   "]}";
            AppendUserMessage(conv, json_msg.c_str());
        }

        LiteRtLm_SamplingParams params = {};
        params.max_tokens = 512;
        params.temperature = 0.7f;

        std::cout << "AI >> " << std::flush;
        g_IsDone = false;
        RunInference(conv, params, MyCallback, nullptr);

        WaitUntilDone(engine, 600);
    }

    DestroyConversation(conv);
    DestroyEngine(engine);
    FreeLibrary(hDll);
    
    // 全局销毁 GDI+ 资源
    Gdiplus::GdiplusShutdown(gdiplusToken);
    return 0;
}
