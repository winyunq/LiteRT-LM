// Copyright (c) 2025-2026 Winyunq. All rights reserved.
//
// Consumer view of LiteRT-LM's complete upstream C API.  The declarations are
// kept in c/engine.h so updating LiteRT-LM updates this surface mechanically.
#ifndef LITERT_LM_RUNTIME_ENGINE_LITERT_LM_NATIVE_API_H_
#define LITERT_LM_RUNTIME_ENGINE_LITERT_LM_NATIVE_API_H_

#if defined(_WIN32) && !defined(LITERT_LM_NATIVE_API_IMPLEMENTATION)
#define LITERT_LM_C_API_EXPORT __declspec(dllimport)
#define LITERT_LM_NATIVE_API_DEFINED_EXPORT 1
#endif

#include "c/engine.h"

#if defined(LITERT_LM_NATIVE_API_DEFINED_EXPORT)
#undef LITERT_LM_NATIVE_API_DEFINED_EXPORT
#undef LITERT_LM_C_API_EXPORT
#endif

#endif  // LITERT_LM_RUNTIME_ENGINE_LITERT_LM_NATIVE_API_H_
