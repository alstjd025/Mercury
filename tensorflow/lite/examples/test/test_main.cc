/* Copyright Minsung Kim, NXC, SNU
   alstjd025@gmail.com
*/
#include <cstdio>
#include <iostream>
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

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "test <tflite model>, <GPU 0/1>\n");
    return 1;
  }
  const char* filename = argv[1];
  bool use_gpu = (atoi(argv[2]) == 0) ? false : true ;

  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
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
    std::cout << "========== GPU delegation done ==========" << "\n";
  }
  // printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  // printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  std::cout << "Test application terminated" << "\n"; 
  return 0;
}
