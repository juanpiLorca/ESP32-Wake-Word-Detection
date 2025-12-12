#include <Arduino.h>
#undef DEFAULT
#include "NeuralNetwork.h"

#include "silence_features_data_quant_int8.h"
#include "unknown_features_data_quant_int8.h"
#include "up_features_data_quant_int8.h"
#include "down_features_data_quant_int8.h"

NeuralNetwork *nn;

// A small struct to organize each test case
struct TestSample {
  const char *name;
  const int8_t *data;
};

void setup() {
  Serial.begin(115200);
  nn = new NeuralNetwork();

  size_t used_bytes = nn->usedBytes();
  Serial.printf("Neural Network initialized. Used bytes: %d\n\n", used_bytes);
}

void runInference(const TestSample &sample) {
  Serial.printf("Running inference for '%s'...\n", sample.name);

  TfLiteTensor *input_tensor = nn->getInputTensor();
  const size_t size = input_tensor->bytes;

  // Copy input feature data into TFLM tensor
  for (size_t i = 0; i < size; i++) {
    input_tensor->data.int8[i] = sample.data[i];
  }

  nn->runInference();

  const int8_t *scores = nn->getQuantizedOutputBuffer();

  Serial.printf(
      "Scores → Silence: %d, Unknown: %d, Up: %d, Down: %d\n\n",
      scores[nn->kSilenceIndex],
      scores[nn->kUnknownIndex],
      scores[nn->kUpIndex],
      scores[nn->kDownIndex]
  );
}

void loop() {

  TestSample tests[] = {
      {"silence", g_silence_features_data_quant_int8},
      {"unknown", g_unknown_features_data_quant_int8},
      {"up",      g_up_features_data_quant_int8},
      {"down",    g_down_features_data_quant_int8},
  };

  const int NUM_TESTS = sizeof(tests) / sizeof(TestSample);

  for (int i = 0; i < NUM_TESTS; i++) {
    runInference(tests[i]);
    delay(2000); 
  }

  Serial.println("=====> Cycle complete. Restarting in 5 seconds <=====");
  Serial.println();
  delay(5000);
}



void showDims() {
  TfLiteIntArray *input_dims = nn->getInputTensor()->dims;
  Serial.printf("Input size: %d, dims->data[0]: %d, dims->data[1]: %d, dims->data[2]: %d, dims->data[3]: %d\n",
                input_dims->size, input_dims->data[0], input_dims->data[1], input_dims->data[2], input_dims->data[3]);
  TfLiteIntArray *output_dims = nn->getOutputTensor()->dims;
  Serial.printf("Output size: %d, dims->data[0]: %d, dims->data[1]: %d\n",
                output_dims->size, output_dims->data[0], output_dims->data[1]);
}