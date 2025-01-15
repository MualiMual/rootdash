import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
import time

# Load the Edge TPU model and delegate
try:
    interpreter = tflite.Interpreter(
        model_path="ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
    )
    interpreter.allocate_tensors()
    print("Edge TPU Delegate loaded successfully.")
except Exception as e:
    print(f"Error loading Edge TPU delegate: {e}")
    exit(1)

# Load COCO labels
with open("coco_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Set up camera
cap = cv2.VideoCapture("/dev/video0")  # Change to '/dev/video1' if necessary
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get model input details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

print("Starting real-time person detection...")

try:
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize and preprocess the frame to match model input size
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (input_shape[2], input_shape[1]))
        input_data = np.expand_dims(img_resized, axis=0).astype(np.uint8)

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]    # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # Class IDs
        scores = interpreter.get_tensor(output_details[2]['index'])[0]   # Confidence scores

        # Loop over all detections and draw bounding boxes for people
        for i in range(len(scores)):
            if scores[i] > 0.5 and int(classes[i]) == 0:  # COCO class ID 0 corresponds to "person"
                ymin, xmin, ymax, xmax = boxes[i]
                x1, y1, x2, y2 = int(xmin * frame.shape[1]), int(ymin * frame.shape[0]), int(xmax * frame.shape[1]), int(ymax * frame.shape[0])

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Person: {int(scores[i] * 100)}%"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show the frame with detection boxes
        cv2.imshow('Real-time Person Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
