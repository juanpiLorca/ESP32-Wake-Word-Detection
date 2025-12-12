#ifndef __NeuralNetwork__
#define __NeuralNetwork__

#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <stdint.h>

namespace tflite
{
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} // namespace tflite

struct TfLiteTensor;

class NeuralNetwork
{
private:
    tflite::MicroMutableOpResolver<20> *resolver;
    tflite::ErrorReporter *error_reporter;
    const tflite::Model *model;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;

    // Setting up interpreter and working memory area
    static constexpr int kArenaSize = 70 * 1024; // 100KB
    alignas(16) static uint8_t tensor_arena[kArenaSize];

public:
    const int kSilenceIndex = 0;
    const int kUnknownIndex = 1;
    const int kDownIndex = 2;
    const int kUpIndex = 3;

    NeuralNetwork();

    TfLiteTensor* getInputTensor() { return input; };
    TfLiteTensor* getOutputTensor() { return output; };

    size_t usedBytes();
    void runInference();
    int8_t *getQuantizedOutputBuffer(); 
};

#endif