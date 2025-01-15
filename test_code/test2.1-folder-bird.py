import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os

# Define the path to your model, labels, and directory of bird photos
MODEL_PATH = "test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite"
LABELS_PATH = "test_data/inat_bird_labels.txt"
IMAGE_DIR = "test_data/bird_photos"  # Directory with bird photos

# Load the Edge TPU delegate and the model
try:
    interpreter = tflite.Interpreter(
        model_path=MODEL_PATH,
        experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    print("Edge TPU Delegate loaded successfully.")
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    exit(1)

# Load the labels
with open(LABELS_PATH, "r") as f:
    labels = f.readlines()

# Set up input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Loop through each image in the directory
for image_name in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    # Ensure we're only processing images
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    print(f"\nProcessing image: {image_name}")

    # Load and preprocess the image
    image = Image.open(image_path).resize((224, 224))
    input_data = np.expand_dims(image, axis=0).astype(np.uint8)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor and display the result
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(output_data)
    
    # Print the prediction result for this image
    print(f"Predicted: {labels[predicted_label].strip()}, Confidence: {output_data[0][predicted_label]}")
