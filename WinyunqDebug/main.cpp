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
typedef void (*PN_AppendUserMessage)(void* conv_ptr, const char* text);
typedef void (*PN_AppendMessageJson)(void* conv_ptr, const char* json_msg);
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

std::string ResizeImageGDI(const std::string& inputPath, int maxDim = 512) {
    // 初始化 GDI+
    ULONG_PTR gdiplusToken;
    Gdiplus::GdiplusStartupInput gdiplusStartupInput;
    Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

    std::string outputPath = "D:\\LiteRT-LM\\WinyunqDebug\\winyunq_temp_multimodal.png";
    bool success = false;
    {
        // 转换宽字符路径
        std::wstring wInputPath(inputPath.begin(), inputPath.end());
        std::wstring wOutputPath(outputPath.begin(), outputPath.end());

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
    Gdiplus::GdiplusShutdown(gdiplusToken);

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

    HMODULE hDll = LoadLibraryA("litert_lm_wrapper.dll");
    if (!hDll) {
        std::cerr << "Failed to load DLL. Error: " << GetLastError() << std::endl;
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
    auto AppendMessageJson = (PN_AppendMessageJson)GetProcAddress(hDll, "LiteRtLm_AppendMessageJson");
    auto RunInference = (PN_RunInference)GetProcAddress(hDll, "LiteRtLm_RunInference");
    auto WaitUntilDone = (PN_WaitUntilDone)GetProcAddress(hDll, "LiteRtLm_WaitUntilDone");

    if (!CreateEngine || !RunInference || !WaitUntilDone || !AppendMessageJson) {
        std::cerr << "Failed to resolve symbols." << std::endl;
        return 1;
    }

    // 1. Initialize Engine
    LiteRtLm_Config config = {};
    config.model_path = "D:\\gemma-4-E4B-it.litertlm";
    config.backend = "gpu";
    config.max_num_tokens = 2048;
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
                if (std::filesystem::exists(path)) {
                    hasImage = true;
                } else {
                    std::cout << "[Warning: Image file not found at \"" << path << "\", falling back to pure text mode]" << std::endl;
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
            
            AppendMessageJson(conv, json_msg.c_str());
        } else {
            AppendUserMessage(conv, input.c_str());
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
    return 0;
}
