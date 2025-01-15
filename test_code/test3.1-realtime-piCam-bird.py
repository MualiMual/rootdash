import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from PIL import Image
from picamera2 import Picamera2
import time

# Load the Edge TPU model and delegate
try:
    interpreter = tflite.Interpreter(
        model_path="mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    print("Edge TPU Delegate loaded successfully.")
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    exit(1)

# Load labels
with open("test_data/inat_bird_labels.txt", "r") as f:
    labels = f.readlines()

# Set up Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# Get input details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print("Starting real-time object detection with Picam...")

try:
    while True:
        # Capture frame-by-frame from Picamera2
        frame = picam2.capture_array()
        
        # Resize frame to match model input size
        img = cv2.resize(frame, (224, 224))
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
        cv2.imshow('Real-time Detection with Picam', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

# Release the camera and close windows
picam2.close()
cv2.destroyAllWindows()
