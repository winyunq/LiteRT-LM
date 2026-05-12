#include <stdio.h>
#include <windows.h>
#include "litert_lm_wrapper.h"

void OnChunk(const char* text, void* user_data) {
    printf("%s", text);
    fflush(stdout);
}

void OnDone(float time_ms, int tokens, float speed, void* user_data) {
    printf("\n[Done] Speed: %.2f tokens/sec\n", speed);
}

int main() {
    const char* model = "D:\\gemma-4-E4B-it.litertlm";
    printf("Initializing Engine...\n");
    int res = LiteRtLm_InitEngine(model, "gpu");
    if (res != 0) {
        printf("Failed to init engine: %d\n", res);
        return -1;
    }

    // 模拟 Agent A
    printf("\n--- Agent A: First Turn ---\n");
    LiteRtLm_ChatCompletion("AgentA", "{\"role\":\"user\",\"content\":\"Hi, I am Alice.\"}", OnChunk, OnDone, NULL);
    Sleep(5000); // 简单等待生成

    // 模拟切换到 Agent B
    printf("\n--- Agent B: First Turn ---\n");
    LiteRtLm_ChatCompletion("AgentB", "{\"role\":\"user\",\"content\":\"Hi, I am Bob.\"}", OnChunk, OnDone, NULL);
    Sleep(5000);

    // 再次切回 Agent A (验证缓存命中)
    printf("\n--- Agent A: Second Turn (Should be fast) ---\n");
    LiteRtLm_ChatCompletion("AgentA", "{\"role\":\"user\",\"content\":\"Do you remember my name?\"}", OnChunk, OnDone, NULL);
    Sleep(5000);

    LiteRtLm_UninitEngine();
    return 0;
}
