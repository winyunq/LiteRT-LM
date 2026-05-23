// Copyright (c) 2026 Winyunq. All rights reserved.
#include <iostream>
#include <string>
#include <windows.h>

#include "../runtime/engine/litert_lm_wrapper.h"

typedef void* (*PN_CreateEngine)(LiteRtLm_Config config);
typedef void (*PN_DestroyEngine)(void* engine_ptr);

int main(int argc, char* argv[]) {
    std::cout << "======================================================================" << std::endl;
    std::cout << "  LiteRT-LM Wrapper DLL 交互式 [显存物理退还] 与生命周期测试 Demo" << std::endl;
    std::cout << "======================================================================" << std::endl;

    const char* model_path = "D:\\UE5Project\\FabUMGMCP\\plugins\\FabUmgMcp\\Resources\\Models\\gemma-4-E2B-it.litertlm";
    if (argc > 1) {
        model_path = argv[1];
    }

    std::cout << "[Config] Model Path: " << model_path << std::endl;
    std::cout << "[Config] Backend: gpu (WebGPU/DirectML)" << std::endl;
    std::cout << "[Config] Tip: 请打开任务管理器(Performance -> GPU -> Dedicated GPU Memory)实时观测显存量变化。" << std::endl;

    HMODULE hDll = nullptr;
    PN_CreateEngine CreateEngine = nullptr;
    PN_DestroyEngine DestroyEngine = nullptr;
    void* active_engine = nullptr;

    LiteRtLm_Config config = {};
    config.model_path = model_path;
    config.backend = "gpu";
    config.max_num_tokens = 128000;
    config.num_threads = 8;
    config.bOptimizeShader = 1;
    config.bEnableVision = 1;
    config.prefill_chunk_size = 32000;

    std::string choice;
    while (true) {
        std::cout << "\n======================================================================" << std::endl;
        std::cout << "  >>> 请选择操作 (请输入字母并回车) <<<" << std::endl;
        std::cout << "  [L] : 【加载模型】(Load Model) -> 分配 D3D12/WebGPU 设备，显存开始强占" << std::endl;
        std::cout << "  [U] : 【物理卸载】(Unload Model) -> 释放 GPU 环境句柄与 DLL，显存物理归还" << std::endl;
        std::cout << "  [Q] : 【退出程序】(Quit)" << std::endl;
        std::cout << "======================================================================" << std::endl;
        std::cout << "您的输入: ";
        std::cin >> choice;

        if (choice == "l" || choice == "L") {
            if (active_engine) {
                std::cout << "[!] 模型已在运行中，请勿重复加载！" << std::endl;
                continue;
            }

            // 1. 动态加载 DLL
            if (!hDll) {
                std::cout << "[Step 1] Loading wrapper DLL (litert_lm_wrapper.dll)..." << std::endl;
                hDll = LoadLibraryA("litert_lm_wrapper.dll");
                if (!hDll) {
                    std::cerr << "[-] 失败：无法载入 DLL！错误代码: " << GetLastError() << std::endl;
                    std::cerr << "[-] 请确认当前目录下已准备好 litert_lm_wrapper.dll 及其加速器依赖库。" << std::endl;
                    continue;
                }
                std::cout << "[+] DLL 载入成功。" << std::endl;

                CreateEngine = (PN_CreateEngine)GetProcAddress(hDll, "LiteRtLm_CreateEngine");
                DestroyEngine = (PN_DestroyEngine)GetProcAddress(hDll, "LiteRtLm_DestroyEngine");

                if (!CreateEngine || !DestroyEngine) {
                    std::cerr << "[-] 失败：未能在 DLL 中定位 CreateEngine 或 DestroyEngine 函数指针。" << std::endl;
                    FreeLibrary(hDll);
                    hDll = nullptr;
                    continue;
                }
            }

            // 2. 调用 CreateEngine 加载模型
            std::cout << "[Step 2] Calling LiteRtLm_CreateEngine..." << std::endl;
            std::cout << "[Wait] 正在初始化 WebGPU 加速器与载入模型，显存即将涨起，请稍候..." << std::endl;
            active_engine = CreateEngine(config);

            if (!active_engine) {
                std::cerr << "[-] 失败：大模型底座引擎创建返回 nullptr！请检查模型路径或 GPU 环境。" << std::endl;
            } else {
                std::cout << "[++++] 成功：模型加载完毕！引擎句柄: " << active_engine << std::endl;
                std::cout << "[++++] 【请观测】任务管理器中的专用显存(Dedicated GPU Memory)现在应该已明显涨起！" << std::endl;
            }
        }
        else if (choice == "u" || choice == "U") {
            if (!hDll || !DestroyEngine) {
                std::cout << "[!] 当前没有加载模型或 DLL，无需重复卸载！" << std::endl;
                continue;
            }

            // 无论 active_engine 是否为 nullptr，都必须允许用户强行执行物理逆向卸载！
            // 因为在 CreateEngine 内部执行失败时，底层的 WebGPU 设备和 shader 编译可能已经成功在 GPU 上分配并强占了显存，
            // 此时必须允许通过 DestroyEngine(nullptr) 或直接清理静态环境及释放 DLL 来退还这部分显存！

            // 1. 调用 DestroyEngine 彻底释放大模型和全局静态环境句柄
            std::cout << "[Step 1] Calling LiteRtLm_DestroyEngine..." << std::endl;
            std::cout << "[Wait] 正在清空全局静态 GPU 环境缓存以释放底层设备，请稍候..." << std::endl;
            DestroyEngine(active_engine);
            active_engine = nullptr;
            std::cout << "[+] 底座引擎已销毁，底座 GPU 环境句柄已完全析构。" << std::endl;

            // 2. 强力物理释放 DLL
            if (hDll) {
                std::cout << "[Step 2] Releasing DLL handle with FreeLibrary..." << std::endl;
                // 我们在 Windows 系统上重复调用 FreeLibrary，以清除多模态在其它地方被隐式引入的引用计数，彻底触发 DLL 的完全脱离
                int count = 0;
                while (FreeLibrary(hDll) && count < 10) {
                    count++;
                }
                hDll = nullptr;
                CreateEngine = nullptr;
                DestroyEngine = nullptr;
                std::cout << "[+] DLL 物理卸载完成。" << std::endl;
            }

            std::cout << "[----] 物理级逆向卸载完毕！" << std::endl;
            std::cout << "[----] 【请观测】任务管理器中的 GPU 专用显存现在应该已瞬间断崖式跌落回零（或回落至基线）！" << std::endl;
        }
        else if (choice == "q" || choice == "Q") {
            // 安全退出前清理资源
            if (active_engine && DestroyEngine) {
                DestroyEngine(active_engine);
            }
            if (hDll) {
                FreeLibrary(hDll);
            }
            std::cout << "[Exit] 测试结束，程序退出。" << std::endl;
            break;
        }
        else {
            std::cout << "[!] 无效的输入，请输入 L（加载）、U（卸载）或 Q（退出）。" << std::endl;
        }
    }

    return 0;
}
