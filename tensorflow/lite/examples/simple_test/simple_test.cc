/* Copyright Minsung Kim, NXC, SNU
   alstjd025@gmail.com
*/
#include <cstdio>
#include <iostream>
#include <ctime>  // clock_gettime
#include <chrono> // optional
#include <fstream>
#include <unistd.h>
#include "tensorflow/lite/core/interpreter_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/optional_debug_tools.h"

// For colors
#define RESET       "\033[0m"
#define BLACK       "\033[30m"
#define RED         "\033[31m"
#define GREEN       "\033[32m"
#define YELLOW      "\033[33m"
#define BLUE        "\033[34m"
#define MAGENTA     "\033[35m"
#define CYAN        "\033[36m"
#define WHITE       "\033[37m"

/* Note: Minsung cmt. No GPU usage. */
// #include "tensorflow/lite/delegates/gpu/delegate.h"

/* Note: Change this value to change the maximum threads to use for inference.*/
#define MAX_THREAD 2

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  size_t res_kb_model_load;
  size_t res_kb_allocation;
  size_t res_kb_delegation;
  size_t res_kb_inference;
  size_t pss_kb_model_load;
  size_t pss_kb_allocation;
  size_t pss_kb_delegation;
  size_t pss_kb_inference;
  if (argc != 4) {
    std::cout << "./simple_test <tflite model>, <inference time N> <print verbose 0/1>" << "\n";
    fprintf(stderr, "Need more aurguments");
    return 1;
  }
  const char* filename = argv[1];
  int inference_time = atoi(argv[2]);
  bool print_verbose = atoi(argv[3]);
  std::cout << GREEN << "LiteRT minimal example start at PID: ";
  std::cout << RED << getpid() << GREEN << "\n"; 
  std::cout << "Model: " << filename << "\n";
  std::cout << "Inference time: " << inference_time << RESET << "\n";
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);

  TFLITE_MINIMAL_CHECK(model != nullptr);

  /* Note: Minsung cmt (not using XNNPACK as default delegate)
           If you wanna uss XNNPACK as CPU backend, uncomment the code below(line 93)
           and comment other(line 94).
  */
  // tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  std::cout << "========== Interpreter build ==========" << "\n";
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  std::cout << "========== Allocated tensors ==========" << "\n";

  /*
    Note: Minsung cmt (turning off GPU delegate)
    =======================================
    TfLiteDelegate* gpu_delegate = nullptr;
    TfLiteGpuDelegateOptionsV2 options = {
      .is_precision_loss_allowed = -1,
      .inference_preference =
            //  TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
          TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
          // TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED,
      .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
      .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
      // .experimental_flags = 1,
      .max_delegated_partitions = 1,
    };
    gpu_delegate = TfLiteGpuDelegateV2Create(&options);
    if(interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk){
      std::cout << "test_main::ModifyGraphWithDelegate() returned ERROR" << "\n";
      return -1;
    }
    std::cout << GREEN << "========== GPU delegation done ==========" 
              << RESET << "\n";
    =======================================
  */
  
  if(print_verbose){
    std::cout << RED << "========== Pre Invoke Interpreter State ==========" 
              << RESET << "\n";
    tflite::PrintInterpreterStateSimple(interpreter.get());
  }
  TFLITE_MINIMAL_CHECK(interpreter->SetNumThreads(MAX_THREAD) == kTfLiteOk);

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`
  // const std::vector<int>& input_tensors = interpreter->inputs();

  // Infernence time measure
  struct timespec start_time, end_time;

  // Run inference!
  std::cout << RED << "Start inference for " << GREEN << inference_time
            << RED << " times" << RESET << "\n"; 
  clock_gettime(CLOCK_MONOTONIC, &start_time);
  for(int i = 0; i<inference_time; ++i){
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  }
  clock_gettime(CLOCK_MONOTONIC, &end_time);
  double total_sec = (end_time.tv_sec - start_time.tv_sec) + 
                    (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
  double average_sec = total_sec / inference_time;

  if(print_verbose){
    std::cout << RED << "========== Post Invoke Interpreter State ==========" 
              << RESET << "\n";
    tflite::PrintInterpreterStateSimple(interpreter.get());
  }
  // Time print
  std::cout << CYAN << "Total Inference Time: " << total_sec << " seconds"
             << "\n";
  std::cout << "Average Inference Time: " << average_sec * 1000 << " ms" 
            << RESET <<  "\n"; 
  
  std::cout << "Test application terminated" << "\n";
  return 0;
}
