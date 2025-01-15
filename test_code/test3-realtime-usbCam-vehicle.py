import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from PIL import Image
import time

# Load the Edge TPU model and delegate for vehicles
try:
    interpreter = tflite.Interpreter(
        model_path="ssd_mobilenet_v2_vehicle_quant_postprocess_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    print("Edge TPU Delegate loaded successfully for vehicle detection.")
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    exit(1)

# Load labels for vehicles
with open("test_data/vehicle_labels.txt", "r") as f:
    labels = f.readlines()

cap = cv2.VideoCapture("/dev/video0")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print("Starting real-time vehicle detection...")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        img = cv2.resize(frame, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(output_data)
        confidence = output_data[0][predicted_label]

        label = labels[predicted_label].strip()
        cv2.putText(frame, f"{label} ({confidence})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
