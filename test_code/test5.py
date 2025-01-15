import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import set_input
from pycoral.adapters.detect import get_objects

# Load the Edge TPU interpreter with the object detection model
model_path = "ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
label_file = "coco_labels.txt"

interpreter = tflite.Interpreter(
    model_path=model_path,
    experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
)
interpreter.allocate_tensors()

# Load labels for object detection
with open(label_file, 'r') as f:
    labels = {int(i.split()[0]): i.split()[1] for i in f.readlines()}

# Video capture from USB camera (adjust /dev/video0 if needed)
cap = cv2.VideoCapture('/dev/video0')

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Function to process each frame and run object detection
def run_inference(frame):
    # Preprocess the frame for input
    resized_frame = cv2.resize(frame, (300, 300))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)

    # Set input tensor
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    output_details = interpreter.get_output_details()
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])

    return boxes, classes, scores

# Main loop for real-time detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    boxes, classes, scores = run_inference(frame)

    # Draw detection boxes and labels on the frame
    for i in range(len(scores[0])):
        if scores[0][i] > 0.5:  # Confidence threshold
            box = boxes[0][i]
            class_id = int(classes[0][i])
            label = labels.get(class_id, "Unknown")

            # Get box coordinates and draw rectangle
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * frame.shape[1])
            xmax = int(xmax * frame.shape[1])
            ymin = int(ymin * frame.shape[0])
            ymax = int(ymax * frame.shape[0])

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"{label}: {scores[0][i]:.2f}", (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the frame with detections
    cv2.imshow("Object Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
