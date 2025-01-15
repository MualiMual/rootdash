import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Paths
model_path = "test_data/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"  # SSD MobileNet V2 model
image_path = "test_data/car.jpg"  # Replace with your image path

# Load the Edge TPU delegate
try:
    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    print("Edge TPU Delegate loaded successfully.")
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    exit(1)

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit(1)

# Preprocess the image
input_data = cv2.resize(image, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
input_data = np.expand_dims(input_data, axis=0)
input_data = input_data.astype(np.uint8)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get the results
boxes = interpreter.get_tensor(output_details[0]['index'])
labels = interpreter.get_tensor(output_details[1]['index'])
scores = interpreter.get_tensor(output_details[2]['index'])

# Print detected objects
for i in range(len(scores[0])):
    if scores[0][i] > 0.5:  # Confidence threshold
        print(f"Detected: Class {int(labels[0][i])} with confidence {scores[0][i]:.2f}")