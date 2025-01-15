import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

# Load the Edge TPU delegate
try:
    delegate = tflite.load_delegate('libedgetpu.so.1')
except Exception as e:
    print(f"Failed to load Edge TPU delegate: {e}")
    delegate = None

# Initialize the interpreter with or without the Edge TPU delegate
if delegate:
    interpreter = tflite.Interpreter(
        model_path='mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite',
        experimental_delegates=[delegate]
    )
else:
    interpreter = tflite.Interpreter(
        model_path='mobilenet_v2_1.0_224_inat_bird_quant.tflite'
    )

interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the image using OpenCV and resize it
image = cv2.imread('parrot.jpg')
if image is None:
    raise ValueError("Image not found or failed to load.")

# Resize and preprocess the image
image_resized = cv2.resize(image, (224, 224))
input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Extract the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Load the labels
with open("inat_bird_labels.txt", "r") as f:
    labels = f.readlines()

# Find the index of the highest confidence score
predicted_index = np.argmax(output_data)

# Print the label of the predicted class
print(f"Predicted: {labels[predicted_index].strip()}, Confidence: {output_data[0][predicted_index]}")
