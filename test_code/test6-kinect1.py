import freenect
import numpy as np
import cv2
from pycoral.adapters import detect
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

# Load the Coral TPU model
model_path = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

# Define a function to get the video frame from Kinect
def get_video():
    frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Define a function to get the depth frame from Kinect
def get_depth():
    depth, _ = freenect.sync_get_depth()
    depth = depth.astype(np.uint8)
    return depth

# Initialize the video window
cv2.namedWindow('Kinect Real-Time Display', cv2.WINDOW_NORMAL)

try:
    while True:
        # Get video frame from Kinect
        frame = get_video()

        # Resize and preprocess the frame for TPU
        input_shape = common.input_size(interpreter)
        resized_frame = cv2.resize(frame, input_shape)
        resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

        # Set the TPU interpreter input
        common.set_input(interpreter, resized_frame_rgb)

        # Run inference
        interpreter.invoke()
        objects = detect.get_objects(interpreter, 0.4)  # Confidence threshold

        # Draw bounding boxes for detected objects
        for obj in objects:
            bbox = obj.bbox
            # Scale bounding box back to original frame size
            x0, y0, x1, y1 = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
            x0 = int(x0 * frame.shape[1] / input_shape[0])
            y0 = int(y0 * frame.shape[0] / input_shape[1])
            x1 = int(x1 * frame.shape[1] / input_shape[0])
            y1 = int(y1 * frame.shape[0] / input_shape[1])

            # Draw the bounding box and label
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            label = f'{obj.id} ({obj.score:.2f})'
            cv2.putText(frame, label, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the combined frame in real-time
        cv2.imshow('Kinect Real-Time Display', frame)

        # Depth map display in a separate window (optional)
        depth_frame = get_depth()
        cv2.imshow('Depth Map', depth_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    freenect.sync_stop()
