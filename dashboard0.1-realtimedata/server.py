from flask import Flask, render_template, jsonify, Response
from collections import deque
import random
import time
from models.models import load_models  # Import from models module
from camera import get_camera
from models.object_detection import generate_frames as generate_frames_with_detection  # Import object detection

app = Flask(__name__)

# Global variable to store the last 5 inference results
last_detections = deque(maxlen=5)

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

# Load models and labels
interpreters, labels = load_models(models)

# Initialize camera
camera = get_camera()

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sensor_data")
def sensor_data():
    """Simulate sensor data."""
    sensor_data = {
        "analog_value": random.randint(0, 1023),
        "color_red": random.randint(0, 255),
        "accel_x": random.uniform(-10, 10),
        "pressure": random.uniform(900, 1100),
        "temperature_sht": random.uniform(10, 30),
    }
    system_stats = {
        "cpu_usage": random.uniform(0, 100),
        "ram_usage": random.uniform(0, 100),
        "storage_usage": random.uniform(0, 100),
        "ip_address": "192.168.1.1",
    }
    return jsonify({**sensor_data, **system_stats})

@app.route("/inference_data")
def inference_data():
    """Return the last 5 inference results."""
    global last_detections
    return jsonify(list(last_detections))

@app.route("/growth_graph")
def growth_graph():
    """Simulate a growth graph."""
    import matplotlib.pyplot as plt
    import io
    import base64

    plt.figure(figsize=(6, 4))
    plt.plot([10, 15, 20, 25, 30], label="Plant A")
    plt.plot([5, 10, 15, 20, 25], label="Plant B")
    plt.plot([8, 12, 16, 20, 24], label="Plant C")
    plt.title("Plant Growth Over Time")
    plt.xlabel("Time (Days)")
    plt.ylabel("Height (cm)")
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return jsonify({"image": image_base64})

@app.route("/seasonal_status")
def seasonal_status():
    """Return seasonal status data."""
    seasonal_status_data = [
        {"plant_name": "Tomato", "start_date": "2023-09-01", "harvest_date": "2023-11-15", "current_stage": random.randint(30, 70)},
        {"plant_name": "Lettuce", "start_date": "2023-09-10", "harvest_date": "2023-11-01", "current_stage": random.randint(40, 80)},
        {"plant_name": "Pepper", "start_date": "2023-08-20", "harvest_date": "2023-11-10", "current_stage": random.randint(50, 90)},
    ]
    return jsonify(seasonal_status_data)

@app.route("/harvest_scheduler")
def harvest_scheduler():
    """Return harvest scheduler data."""
    harvest_scheduler_data = [
        {"plant_name": "Tomato", "predicted_harvest_date": "2023-11-15"},
        {"plant_name": "Lettuce", "predicted_harvest_date": "2023-11-01"},
        {"plant_name": "Pepper", "predicted_harvest_date": "2023-11-10"},
    ]
    return jsonify(harvest_scheduler_data)

@app.route("/video_feed")
def video_feed():
    """Route to stream video from the USB camera with object detection."""
    return Response(generate_frames_with_detection(camera, interpreters, labels, last_detections), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8086)