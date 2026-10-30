package(default_visibility = ["//visibility:public"])

cc_library(
    name = "litert_lm_api_headers",
    hdrs = ["litert_lm_api.h", "litert_lm.hpp"],
)

cc_library(
    name = "litert_lm_api_header_compile_test",
    testonly = 1,
    srcs = ["litert_lm_api_header_c_test.c", "litert_lm_api_header_cpp_test.cc"],
    deps = [":litert_lm_api_headers"],
)

cc_test(
    name = "litert_lm_cpp_facade_test",
    srcs = ["litert_lm_cpp_facade_test.cc"],
    deps = [":litert_lm_api_headers", "@com_google_googletest//:gtest_main"],
)

cc_binary(
    name = "litert_lm_wrapper",
    srcs = ["litert_lm_api.cc", "litert_lm_api.h", "litert_lm.hpp"],
    additional_linker_inputs = select({
        "@platforms//os:android": [":litert_lm_wrapper.lds", "@litert//litert/c:libLiteRt.so"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    linkopts = select({
        "@platforms//os:android": [
            "-Wl,--version-script=$(location :litert_lm_wrapper.lds)",
            "-Wl,--no-as-needed",
            "$(location @litert//litert/c:libLiteRt.so)",
            "-Wl,--as-needed",
        ],
        "//conditions:default": [],
    }),
    win_def_file = "litert_lm_wrapper.def",
    deps = [
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_impl_selected",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:io_types",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/time",
        "@nlohmann_json//:json",
        "//runtime/components/constrained_decoding:llg_constraint_config",
        "//runtime/conversation",
        "//runtime/conversation:thinking_config",
        "//runtime/executor:llm_executor_settings",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:logging",
        "//runtime/util:scoped_file",
        "//runtime/util:litert_status_util",
    ],
)
