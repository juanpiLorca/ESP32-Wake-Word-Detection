import tensorflow as tf

import os

model = tf.lite.Interpreter("model_exports/model_quant.tflite")
model.allocate_tensors()

for op in model._get_ops_details():
    print(op["op_name"])

file_size = os.path.getsize("model_exports/model_quant.tflite")
print("Model file size:", file_size, "bytes")
