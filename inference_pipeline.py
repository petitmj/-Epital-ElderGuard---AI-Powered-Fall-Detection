import tensorflow as tf
from tensorflow import keras
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np
from sklearn.ensemble import IsolationForest
import onnxruntime as ort
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import qai_hub as hub
from keras.saving import register_keras_serializable
from keras.losses import MeanSquaredError
import tf2onnx
from collections.abc import Mapping, MutableMapping

# ===========================
# Load or Define Autoencoder
# ===========================
try:
    autoencoder = keras.models.load_model("fall_detection_model.h5")
    print("✅ Loaded existing autoencoder model.")
except:
    print("🚨 No pre-trained model found. Defining and training a new autoencoder...")
    input_dim = 10  # Adjust based on feature size
    input_layer = keras.layers.Input(shape=(input_dim,))
    encoded = keras.layers.Dense(8, activation="relu")(input_layer)
    decoded = keras.layers.Dense(input_dim, activation="sigmoid")(encoded)
    autoencoder = keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.save("fall_detection_model.h5")
    print("✅ New autoencoder model saved.")

# Convert Autoencoder to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
tflite_model = converter.convert()
with open("fall_detection_model.tflite", "wb") as f:
    f.write(tflite_model)
print("✅ Autoencoder successfully converted to TFLite.")

# ===========================
# Train Isolation Forest Model
# ===========================
train_data = np.random.rand(100, 10).astype(np.float32)
model = IsolationForest(contamination=0.05, random_state=42, max_features=10)
model.fit(train_data)

if not hasattr(model, "_max_features"):
    model._max_features = model.max_features_

# Convert Isolation Forest to ONNX
initial_type = [("input", FloatTensorType([None, train_data.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset={"": 15, "ai.onnx.ml": 3})
with open("fall_detection_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
print("✅ IsolationForest successfully converted to ONNX.")

# ===========================
# Run Inference with TFLite Model
# ===========================
interpreter = tf.lite.Interpreter(model_path="fall_detection_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
sample_data = np.random.rand(1, 10).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sample_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("✅ TFLite Model Output:", output)

# ===========================
# Run Inference with ONNX Model
# ===========================
onnx_model_path = "fall_detection_model.onnx"
session = ort.InferenceSession(onnx_model_path)
input_shape = session.get_inputs()[0].shape  # Example: [None, 10]
num_features = input_shape[1]
sample_input = np.random.rand(1, num_features).astype(np.float32)
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: sample_input})
print("ONNX Model Output:", output)

# ===========================
# Optimize ONNX Model for Mobile
# ===========================
optimized_model_path = "fall_detection_model_optimized.onnx"
quantized_model = quantize_dynamic(onnx_model_path, optimized_model_path, weight_type=QuantType.QInt8)
print("✅ ONNX model optimized for mobile")

# ===========================
# Run Optimized ONNX Inference
# ===========================
session = ort.InferenceSession("fall_detection_model_optimized.onnx")
sample_input = np.random.rand(1, num_features).astype(np.float32)
outputs = session.run(None, {"input": sample_input})
print("✅ Inference successful! Output:", outputs)

# ===========================
# Convert and Compile for Qualcomm AI Hub
# ===========================
@register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

custom_objects = {"mse": mse, "MeanSquaredError": MeanSquaredError()}
model = tf.keras.models.load_model("fall_detection_model.h5", custom_objects=custom_objects)
print("✅ Model loaded successfully!")
model.compile(loss=mse, optimizer="adam")
dummy_input = tf.random.normal([1] + list(model.input_shape[1:]))
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(dummy_input.shape, dtype=tf.float32)], opset=13)
onnx.save_model(onnx_model, "fall_detection_model_aihub.onnx")
print("✅ Model converted successfully for AI Hub!")

# Submit Compilation Jobs for AI Hub
compile_job = hub.submit_compile_job(
    model="fall_detection_model_aihub.onnx",
    device=hub.Device("Samsung Galaxy S23 (Family)"),
)
assert isinstance(compile_job, hub.CompileJob)
print("✅ Compilation job submitted for TFLite!")

compile_job = hub.submit_compile_job(
    model="fall_detection_model_aihub.onnx",
    device=hub.Device("Samsung Galaxy S23 (Family)"),
    options="--target_runtime qnn_lib_aarch64_android",
)
assert isinstance(compile_job, hub.CompileJob)
print("✅ Compilation job submitted for QNN Model Library!")

# ===========================
# Run On-Device Inference
# ===========================
sensor_data = np.random.uniform(0, 20, (1, num_features)).astype(np.float32)
inference_job = hub.submit_inference_job(
    model="fall_detection_model_aihub.onnx",
    device=hub.Device("Samsung Galaxy S23 (Family)"),
    inputs={"args_0": [sensor_data]},
)
assert isinstance(inference_job, hub.InferenceJob), "❌ Inference job submission failed!"
on_device_output = inference_job.download_output_data()
assert isinstance(on_device_output, dict), "❌ Unexpected output format from inference job!"
print("✅ On-device inference completed successfully!")
print("🔹 Output Data:", on_device_output)
