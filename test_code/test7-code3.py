import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np

# Load the Edge TPU delegate
delegate = tflite.load_delegate('/usr/lib/x86_64-linux-gnu/libedgetpu.so.1')

# Load model and labels with Edge TPU delegate
interpreter = tflite.Interpreter(
    model_path="mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()
labels = [line.strip() for line in open("inat_bird_labels.txt").readlines()]

# Load and preprocess image
image = Image.open("grace_hopper.bmp").resize((224, 224))
input_data = np.expand_dims(image, axis=0)

# Set input tensor
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], input_data)

# Perform inference
interpreter.invoke()

# Get output tensor
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = labels[np.argmax(output_data)]

print(f"Predicted label: {predicted_label}")
