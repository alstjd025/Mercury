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

/* Note [Minsung]
  ..
*/
// #define InferenceBeforeGPUDelegation
#define MAX_THREAD 2


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

size_t get_current_process_pss_kb() {
  const char* smaps_path = "/proc/self/smaps";
  std::ifstream smaps_file(smaps_path);

  if (!smaps_file.is_open()) {
    std::cerr << "Failed to open " << smaps_path << std::endl;
    return 0;
  }
  std::string line;
  size_t total_pss_kb = 0;
  while (std::getline(smaps_file, line)) {
    if (line.find("Pss:") == 0) {
      std::istringstream iss(line);
      std::string key;
      size_t pss_kb;

      iss >> key >> pss_kb;
      total_pss_kb += pss_kb;
    }
  }
  smaps_file.close();
  return total_pss_kb;
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
  if (argc != 6) {
    std::cout << "test <tflite model>, <GPU 0/1> <inference time N>"
              << "<Print verbose debug msg 0/1> <max_delegate_partition N>" << "\n";
    fprintf(stderr, "Need more aurguments");
    return 1;
  }
  const char* filename = argv[1];
  bool use_gpu = (atoi(argv[2]) == 0) ? false : true ;
  int inference_time = atoi(argv[3]);
  bool print_verbose = (atoi(argv[4]) == 0) ? false : true ;
  int max_partition = atoi(argv[5]);
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
  res_kb_model_load = get_resident_set_size_kb();
  pss_kb_model_load= get_current_process_pss_kb();
  
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
  res_kb_allocation = get_resident_set_size_kb();
  pss_kb_allocation = get_current_process_pss_kb();

  // TFLITE_MINIMAL_CHECK(interpreter->RemoveAllDelegates() == kTfLiteOk);
  // Minsung 
  // Delegate code here (see benchmark_tflite_model.cc/line 1189)
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
    .max_delegated_partitions = max_partition,
  };
  if(use_gpu){
    #ifdef InferenceBeforeGPUDelegation
      std::cout << "Inference for tensor allocation" << "\n";
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
      std::cout << "Inference for tensor allocation done" << "\n";
    #endif

    gpu_delegate = TfLiteGpuDelegateV2Create(&options);
    if(interpreter->ModifyGraphWithDelegate(gpu_delegate) != kTfLiteOk){
      std::cout << "test_main::ModifyGraphWithDelegate() returned ERROR" << "\n";
      return -1;
    }
    std::cout << GREEN << "========== GPU delegation done ==========" 
              << RESET << "\n";
    res_kb_delegation = get_resident_set_size_kb();
    pss_kb_delegation = get_current_process_pss_kb();
  }
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

  // Run inference
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
  #ifdef latency_per_node
    interpreter->PrintLatencyPerNodeOfSubgraph();
  #endif
  
  // Time print
  std::cout << CYAN << "Total Inference Time: " << total_sec << " seconds"
             << "\n";
  std::cout << "Average Inference Time: " << average_sec * 1000 << " ms" 
            << RESET <<  "\n"; 
  res_kb_inference = get_resident_set_size_kb();
  pss_kb_inference = get_current_process_pss_kb();
  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`

auto print_memory = [](const std::string& title, size_t res_kb, size_t pss_kb) {
    std::cout << "\n" << GREEN << "=== " << title << " ===" << RESET << "\n";
    std::cout << "  RSS (RES): " << GREEN << res_kb << " kB" << RESET 
              << " (" << GREEN << (res_kb / 1024.0) << " MB" << RESET << ")" << "\n";
    std::cout << "  PSS      : " << GREEN << pss_kb << " kB" << RESET 
              << " (" << GREEN << (pss_kb / 1024.0) << " MB" << RESET << ")" << "\n";
};

  std::cout << "Test application terminated" << "\n";
  print_memory("Memory after model load", res_kb_model_load, pss_kb_model_load);
  print_memory("Memory after tensor allocation for CPU", res_kb_allocation, pss_kb_allocation);
  if (use_gpu) {
      print_memory("Memory after GPU delegation", res_kb_delegation, pss_kb_delegation);
  }
  print_memory("Memory after inference", res_kb_inference, pss_kb_inference);
  return 0;
}
