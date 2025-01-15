import cv2
import numpy as np
from collections import deque

# Global variable to store the last 5 inference results
last_detections = deque(maxlen=5)

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define models and labels for CORAL object detection
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
    },
    "flowers": {
        "model_path": "test_data/mobilenet_v2_1.0_224_flowers_quant_edgetpu.tflite",
        "label_path": "test_data/flower_labels.txt"
    },
    "animals": {
        "model_path": "test_data/mobilenet_v2_1.0_224_inat_mammal_quant_edgetpu.tflite",
        "label_path": "test_data/inat_mammal_labels.txt"
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

def generate_frames():
    """Generate video frames with object detection."""
    camera = cv2.VideoCapture(0)  # Use the default camera (index 0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Apply background subtraction to detect motion
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around moving objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Process frame with each model
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

            # Get the label
            label = labels[category][predicted_label].strip()

            # Add the result to the last_detections list
            last_detections.append({
                "category": category,
                "label": label,
                "confidence": float(confidence)
            })

            # Display the result on the frame
            cv2.putText(frame, f"{category.upper()}: {label} ({confidence:.2f})",
                        (10, 30 + 40 * list(interpreters.keys()).index(category)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')