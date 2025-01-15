from flask import Flask, render_template, jsonify, Response
from collections import deque
import random
import time

app = Flask(__name__)

# Global variable to store the last 5 inference results
last_detections = deque(maxlen=5)

# Dummy sensor data for testing
def read_sensors():
    """Simulate sensor data."""
    return {
        "analog_value": random.randint(0, 1023),
        "color_red": random.randint(0, 255),
        "accel_x": random.uniform(-10, 10),
        "pressure": random.uniform(900, 1100),
        "temperature_sht": random.uniform(10, 30),
    }

# Dummy system stats for testing
def get_system_stats():
    """Simulate system monitoring data."""
    return {
        "cpu_usage": random.uniform(0, 100),
        "ram_usage": random.uniform(0, 100),
        "storage_usage": random.uniform(0, 100),
        "ip_address": "192.168.1.1",
    }

# Dummy growth graph for testing
def generate_growth_graph():
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
    return image_base64

# Dummy video feed for testing
def generate_frames():
    """Simulate a video feed."""
    import cv2
    import numpy as np

    while True:
        # Create a dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Live Video Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/sensor_data")
def sensor_data():
    sensor_data = read_sensors()
    system_stats = get_system_stats()
    return jsonify({**sensor_data, **system_stats})

@app.route("/inference_data")
def inference_data():
    """Return the last 5 inference results."""
    global last_detections
    # Simulate inference data
    if len(last_detections) == 0:
        last_detections.extend([
            {"category": "plants", "label": "Tomato", "confidence": random.uniform(0, 100)},
            {"category": "bugs", "label": "Ladybug", "confidence": random.uniform(0, 100)},
            {"category": "birds", "label": "Sparrow", "confidence": random.uniform(0, 100)},
            {"category": "flowers", "label": "Rose", "confidence": random.uniform(0, 100)},
            {"category": "animals", "label": "Cat", "confidence": random.uniform(0, 100)},
        ])
    return jsonify(list(last_detections))

@app.route("/growth_graph")
def growth_graph():
    """Return the growth graph as a base64 encoded image."""
    return jsonify({"image": generate_growth_graph()})

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
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8086)