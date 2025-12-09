#include <Arduino.h>

#undef DEFAULT

#include "NeuralNetwork.h"

#include "up_features_data_quant.h"
#include "down_features_data_quant.h"

NeuralNetwork *nn;

void setup()
{
  Serial.begin(115200);
  nn = new NeuralNetwork();

  size_t used_bytes = nn->usedBytes();
  Serial.printf("Neural Network initialized. Used bytes: %d\n", used_bytes);
}

void loop()
{ 

  const uint8_t* up_features_data = g_up_features_data_quant_data; 
  const size_t size = nn->getInputTensor()->bytes;
  TfLiteTensor *input_buff = nn->getInputTensor(); 
  for (int i = 0; i < size; i++) {
    input_buff->data.uint8[i] = up_features_data[i];
  }

  TfLiteIntArray *input_dims = nn->getInputTensor()->dims;
  Serial.printf("Input size: %d, dims->data[0]: %d, dims->data[1]: %d, dims->data[2]: %d, dims->data[3]: %d\n",
                input_dims->size, input_dims->data[0], input_dims->data[1], input_dims->data[2], input_dims->data[3]);

  nn->runInference();
  
  TfLiteIntArray *output_dims = nn->getOutputTensor()->dims;
  Serial.printf("Output size: %d, dims->data[0]: %d, dims->data[1]: %d\n",
                output_dims->size, output_dims->data[0], output_dims->data[1]);

  uint8_t silence_score = nn->getQuantizedOutputBuffer()[nn->kSilenceIndex];
  uint8_t unknown_score = nn->getQuantizedOutputBuffer()[nn->kUnknownIndex];
  uint8_t up_score = nn->getQuantizedOutputBuffer()[nn->kUpIndex];
  uint8_t down_score = nn->getQuantizedOutputBuffer()[nn->kDownIndex];

  Serial.printf("Scores - Silence: %d, Unknown: %d, Up: %d, Down: %d\n",
                silence_score, unknown_score, up_score, down_score);

  delay(5000);
}