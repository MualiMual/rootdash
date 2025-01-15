import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

# Load the Edge TPU delegate
try:
    interpreter = tflite.Interpreter(
        model_path="test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    print("Edge TPU Delegate loaded successfully.")
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    exit(1)

# Load the labels
with open("test_data/inat_bird_labels.txt", "r") as f:
    labels = f.readlines()

# Load and preprocess the image
image = Image.open("test_data/parrot.jpg").resize((224, 224))
input_data = np.expand_dims(image, axis=0).astype(np.uint8)

# Set input tensor
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the inference
interpreter.invoke()

# Get output tensor and display the result
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data)

print(f"Predicted: {labels[predicted_label].strip()}, Confidence: {output_data[0][predicted_label]}")
