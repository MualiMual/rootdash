import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from PIL import Image
import time

# Load the Edge TPU model and delegate
try:
    interpreter = tflite.Interpreter(
        model_path="test_data/ssd_mobilenet_v1_coco_quant_postprocess_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    print("Edge TPU Delegate loaded successfully for people detection.")
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    exit(1)

# Load labels for COCO dataset
with open("test_data/coco_labels.txt", "r") as f:
    labels = f.readlines()

# Set up camera
cap = cv2.VideoCapture("/dev/video0")  # Change to '/dev/video1' if necessary
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get input details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print("Starting real-time people detection...")

try:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame to match model input size
        img = cv2.resize(frame, (input_shape[1], input_shape[2]))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor and process the result
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(output_data)
        confidence = output_data[0][predicted_label]

        # Display the result on the frame
        label = labels[predicted_label].strip()
        cv2.putText(frame, f"{label} ({confidence})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("People Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
