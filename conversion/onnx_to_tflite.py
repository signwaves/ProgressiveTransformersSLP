import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model_path = "sign_language_model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Prepare the ONNX model for TensorFlow
tf_rep = prepare(onnx_model)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_saved_model(tf_rep.tf_module().save())

# Apply quantization (optional but recommended)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# You can also explore different quantization types:
# converter.target_spec.supported_types = [tf.float16]  # For float16 quantization
# or
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8
# and provide a representative dataset

tflite_model = converter.convert()

# Save the TensorFlow Lite model
tflite_model_path = "sign_language_model.tflite"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"Model converted to TensorFlow Lite: {tflite_model_path}")