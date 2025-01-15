import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from PIL import Image
import time

# Define model and label paths for plants, bugs, and birds
models = {
    "plants": {
        "model_path": "test_data/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite",
        "label_path": "test_data/inat_plant_labels.txt"
    },
    "bugs": {
        "model_path": "test_data/mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite",
        "label_path": "test_data/inat_insect_labels.txt"
    },
    "birds": {
        "model_path": "test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite",
        "label_path": "test_data/inat_bird_labels.txt"
    }
}

# Load all interpreters and labels at once
interpreters = {}
labels = {}

for category, paths in models.items():
    try:
        interpreter = tflite.Interpreter(
            model_path=paths["model_path"],
            experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
        )
        interpreter.allocate_tensors()
        interpreters[category] = interpreter

        with open(paths["label_path"], "r") as f:
            labels[category] = f.readlines()

        print(f"Edge TPU Delegate loaded successfully for {category} detection.")
    except Exception as e:
        print(f"Error loading {category} model or labels: {e}")

# Set up camera
cap = cv2.VideoCapture("/dev/video0")  # Change to '/dev/video1' if necessary
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Starting real-time detection...")

# Start detection loop
try:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize frame to match model input size
        for category, interpreter in interpreters.items():
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            input_shape = input_details[0]['shape']

            img = cv2.resize(frame, (input_shape[1], input_shape[2]))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(img_rgb, axis=0).astype(np.uint8)

            # Set input tensor and run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Get the output tensor and process the result
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data)
            confidence = output_data[0][predicted_label]

            # Display the result on the frame
            label = labels[category][predicted_label].strip()
            cv2.putText(frame, f"{category.upper()}: {label} ({confidence:.2f})", 
                        (10, 30 + 40 * list(interpreters.keys()).index(category)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
