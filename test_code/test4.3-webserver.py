from bottle import route, run, response
import tflite_runtime.interpreter as tflite
import numpy as np
import cv2
from io import BytesIO

# Define models and labels
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
            labels[category] = [line.strip() for line in f.readlines()]

        print(f"Edge TPU Delegate loaded successfully for {category} detection.")
    except Exception as e:
        print(f"Error loading {category} model or labels: {e}")

# Set up camera
camera_device = "/dev/video14"  # Change this to the correct device (e.g., /dev/video14)
cap = cv2.VideoCapture(camera_device)

# Set camera format and resolution
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG format
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height

if not cap.isOpened():
    print(f"Error: Could not open camera at {camera_device}.")
    print("Available cameras:")
    import os
    os.system("ls /dev/video*")
    exit()

def get_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None

    # Process frame with each model
    for category, interpreter in interpreters.items():
        try:
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
        except Exception as e:
            print(f"Error processing {category} model: {e}")

    # Encode the frame to JPEG
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes()

@route('/video_feed')
def video_feed():
    response.content_type = 'multipart/x-mixed-replace; boundary=frame'
    return stream_video()

def stream_video():
    while True:
        frame = get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@route('/')
def index():
    return '''
        <html>
            <head>
                <title>Object Detection Stream</title>
                <style>
                    body {
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                        margin: 0;
                        background-color: #333;
                    }
                    img {
                        max-width: 100%;
                        max-height: 100%;
                        border: 5px solid #fff;
                        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.5);
                    }
                </style>
            </head>
            <body>
                <h1 style="color: white;">Object Detection Video Stream</h1>
                <img src="/video_feed" />
            </body>
        </html>
    '''

# Run the Bottle web server
run(host='0.0.0.0', port=8082)
