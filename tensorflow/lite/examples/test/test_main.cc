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
#include "tensorflow/lite/delegates/gpu/delegate.h"


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

size_t get_resident_set_size_kb() {
    std::ifstream status_file("/proc/self/status");
    std::string line;

    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            size_t res_kb;
            sscanf(line.c_str(), "VmRSS: %lu kB", &res_kb);
            return res_kb;
        }
    }
    return 0;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    fprintf(stderr, "test <tflite model>, <GPU 0/1> <inference time N>\n");
    return 1;
  }
  const char* filename = argv[1];
  bool use_gpu = (atoi(argv[2]) == 0) ? false : true ;
  int inference_time = atoi(argv[3]);
  std::cout << GREEN << "LiteRT minimal example start at PID: ";
  std::cout << RED << getpid() << GREEN << "\n"; 
  std::cout << "Model: " << filename << "\n";
  std::cout << "GPU delegation: "; 
  if(use_gpu){
    std::cout << BLUE << "ON" << GREEN << "\n";
  }else{
    std::cout << RED << "OFF" << GREEN << "\n";
  }
  std::cout << "Inference time: " << inference_time << RESET << "\n";
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  size_t res_kb = get_resident_set_size_kb();
  double res_mb = res_kb / 1024.0; // 1MB = 1024kB

  std::cout << RED << "Resident Set Size after model load: " 
            << res_kb << " kB (" 
            << res_mb << " MB)" << RESET << "\n";
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;

  // Minsung disabled (using XNNPACK as default delegate)
  // tflite::ops::builtin::BuiltinOpResolver resolver;


  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  std::cout << "========== Interpreter build ==========" << "\n";
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  std::cout << "========== Allocated tensors ==========" << "\n";
  res_kb = get_resident_set_size_kb();
  res_mb = res_kb / 1024.0; // 1MB = 1024kB
  std::cout << RED << "Resident Set Size after tensor allocation for CPU: " 
            << res_kb << " kB (" 
            << res_mb << " MB)" << RESET << "\n";


  // TFLITE_MINIMAL_CHECK(interpreter->RemoveAllDelegates() == kTfLiteOk);
  // Minsung 
  // Delegate code here (see benchmark_tflite_model.cc/line 1189)
  TfLiteGpuDelegateOptionsV2 options = {
    .is_precision_loss_allowed = 0,
    .inference_preference =
        TFLITE_GPU_INFERENCE_PREFERENCE_FAST_SINGLE_ANSWER,
    .inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY,
    .inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE,
    .inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO,
    // .experimental_flags = 1,
    .max_delegated_partitions = 1000,
  };
  TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&options);
  if(use_gpu){
    if(interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk){
      std::cout << "test_main::ModifyGraphWithDelegate() returned ERROR" << "\n";
      return -1;
    }
    std::cout << GREEN << "========== GPU delegation done ==========" 
              << RESET << "\n";
    size_t res_kb = get_resident_set_size_kb();
    double res_mb = res_kb / 1024.0; // 1MB = 1024kB

    std::cout << RED << "Resident Set Size after GPU delegation: " 
              << res_kb << " kB (" 
              << res_mb << " MB)" << RESET << "\n";
  }
  std::cout << RED << "========== Pre Invoke Interpreter State ==========" 
            << RESET << "\n";
  tflite::PrintInterpreterStateSimple(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // Infernence time measure
  struct timespec start_time, end_time;
  clock_gettime(CLOCK_MONOTONIC, &start_time);

  // Run inference
  for(int i = 0; i<inference_time; ++i){
    TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
  double total_sec = (end_time.tv_sec - start_time.tv_sec) + 
                    (end_time.tv_nsec - start_time.tv_nsec) / 1e9;
  double average_sec = total_sec / inference_time;

  // Time print
  std::cout << CYAN << "Total Inference Time: " << total_sec << " seconds"
             << "\n";
  std::cout << "Average Inference Time: " << average_sec * 1000 << " ms" 
            << RESET <<  "\n"; 

  std::cout << RED << "========== Post Invoke Interpreter State ==========" 
            << RESET << "\n";
  tflite::PrintInterpreterStateSimple(interpreter.get());
  res_kb = get_resident_set_size_kb();
  res_mb = res_kb / 1024.0; // 1MB = 1024kB
  std::cout << RED << "Resident Set Size after inference: " 
            << res_kb << " kB (" 
            << res_mb << " MB)" << RESET << "\n";
  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  std::cout << "Test application terminated" << "\n"; 
  return 0;
}
