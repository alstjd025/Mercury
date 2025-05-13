load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_copts_warnings", "tflite_linkopts")
load("//tensorflow/lite:special_rules.bzl", "tflite_portable_test_suite")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],
)

common_copts = tflite_copts() + tflite_copts_warnings()

# We create a library for benchmark_main.cc to faciliate the creation of a
# customized benchmark model binary that only needs linking with extra
# dependency, e.g., enabling creating of benchmark binaries with a custom
# delegate provider.
cc_library(
    name = "simple_test_lib",
    srcs = [
        "simple_test.cc",
    ],
    copts = common_copts + select({
        "//tensorflow:ios": [
            "-xobjective-c++",
        ],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorflow/core/example:example_protos_cc_impl",
        "//tensorflow/lite:framework",
        "//tensorflow/lite:simple_memory_arena_debug_dump",
        "//tensorflow/lite:string_util",
        "//tensorflow/lite:op_resolver",
        "//tensorflow/lite/core:cc_api_stable",
        "//tensorflow/lite/core:framework",
        "//tensorflow/lite/core/c:c_api_types",
        "//tensorflow/lite/core/c:common",
        "//tensorflow/lite/kernels:builtin_ops",
        "//tensorflow/lite/kernels:cpu_backend_context",
        "//tensorflow/lite/profiling:model_runtime_info",
        "//tensorflow/lite/tools:logging",
        "//tensorflow/lite/tools:model_loader",
        "//tensorflow/lite/tools:utils",
        "//tensorflow/lite/nnapi:nnapi_util",
        # "//tensorflow/lite/tools/delegates:delegate_provider_hdr",
        # "//tensorflow/lite/tools/delegates:tflite_execution_providers",
        "//tensorflow/core/util:stats_calculator_portable",
        # "//tensorflow/lite/delegates/hexagon:hexagon_delegate",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@ruy//ruy/profiler",
    ]+ select({
        "//tensorflow:android": [
            "//tensorflow/lite/delegates/gpu:delegate",
        ],
        "//conditions:default": [],
    }),
)

cc_binary(
    name = "simple_test",
    copts = common_copts,
    linkopts = tflite_linkopts() + select({
        "//tensorflow:android": [
            "-pie",  # Android 5.0 and later supports only PIE
            "-lm",  # some builtin ops, e.g., tanh, need -lm
            "-Wl,--rpath=/data/local/tmp/",  # Hexagon delegate libraries should be in /data/local/tmp
        ],
        "//conditions:default": [],
    }),
    tags = ["builder_default_android_arm64"],
    deps = [
        ":simple_test_lib",
    ],
)


tflite_portable_test_suite()
