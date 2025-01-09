from flask import Flask, render_template, jsonify, Response
from collections import deque
import random
import time
import mariadb
from models.models import load_models  # Import from models module
from utils.camera import get_camera
from models.object_detection import generate_frames as generate_frames_with_detection  # Import object detection

app = Flask(__name__)

# MariaDB configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'rootdash_user'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'rootdash'

# Connect to MariaDB
def get_db_connection():
    try:
        conn = mariadb.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )
        return conn
    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")
        return None

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
    """Simulate sensor data and store it in MariaDB."""
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

    # Insert sensor data into MariaDB
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_data (analog_value, color_red, accel_x, pressure, temperature_sht, cpu_usage, ram_usage, storage_usage, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sensor_data["analog_value"],
            sensor_data["color_red"],
            sensor_data["accel_x"],
            sensor_data["pressure"],
            sensor_data["temperature_sht"],
            system_stats["cpu_usage"],
            system_stats["ram_usage"],
            system_stats["storage_usage"],
            system_stats["ip_address"]
        ))
        conn.commit()
        conn.close()

    return jsonify({**sensor_data, **system_stats})

@app.route("/inference_data")
def inference_data():
    """Return the last 5 inference results."""
    global last_detections
    return jsonify(list(last_detections))

@app.route("/growth_graph")
def growth_graph():
    """Simulate a growth graph and store it in MariaDB."""
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

    # Insert growth graph data into MariaDB
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO growth_graph (image) VALUES (?)
        ''', (image_base64,))
        conn.commit()
        conn.close()

    return jsonify({"image": image_base64})

@app.route("/growth_rate")
def growth_rate():
    """Return growth rate data and store it in MariaDB."""
    growth_rate_data = [
        {"plant_name": "Plant A", "rate": "10 cm", "height": "30 cm", "time_after_planting": "2 weeks"},
        {"plant_name": "Plant B", "rate": "5 cm", "height": "25 cm", "time_after_planting": "3 weeks"},
        {"plant_name": "Plant C", "rate": "8 cm", "height": "24 cm", "time_after_planting": "4 weeks"},
    ]

    # Insert growth rate data into MariaDB
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        for data in growth_rate_data:
            cursor.execute('''
                INSERT INTO growth_rate (plant_name, rate, height, time_after_planting)
                VALUES (?, ?, ?, ?)
            ''', (
                data["plant_name"],
                data["rate"],
                data["height"],
                data["time_after_planting"]
            ))
        conn.commit()
        conn.close()

    return jsonify(growth_rate_data)

@app.route("/seasonal_status")
def seasonal_status():
    """Return seasonal status data and store it in MariaDB."""
    seasonal_status_data = [
        {"plant_name": "Tomato", "start_date": "2023-09-01", "harvest_date": "2023-11-15", "current_stage": random.randint(30, 70)},
        {"plant_name": "Lettuce", "start_date": "2023-09-10", "harvest_date": "2023-11-01", "current_stage": random.randint(40, 80)},
        {"plant_name": "Pepper", "start_date": "2023-08-20", "harvest_date": "2023-11-10", "current_stage": random.randint(50, 90)},
    ]

    # Insert seasonal status data into MariaDB
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        for data in seasonal_status_data:
            cursor.execute('''
                INSERT INTO seasonal_status (plant_name, start_date, harvest_date, current_stage)
                VALUES (?, ?, ?, ?)
            ''', (
                data["plant_name"],
                data["start_date"],
                data["harvest_date"],
                data["current_stage"]
            ))
        conn.commit()
        conn.close()

    return jsonify(seasonal_status_data)

@app.route("/harvest_scheduler")
def harvest_scheduler():
    """Return harvest scheduler data and store it in MariaDB."""
    harvest_scheduler_data = [
        {"plant_name": "Tomato", "predicted_harvest_date": "2023-11-15"},
        {"plant_name": "Lettuce", "predicted_harvest_date": "2023-11-01"},
        {"plant_name": "Pepper", "predicted_harvest_date": "2023-11-10"},
    ]

    # Insert harvest scheduler data into MariaDB
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        for data in harvest_scheduler_data:
            cursor.execute('''
                INSERT INTO harvest_scheduler (plant_name, predicted_harvest_date)
                VALUES (?, ?)
            ''', (
                data["plant_name"],
                data["predicted_harvest_date"]
            ))
        conn.commit()
        conn.close()

    return jsonify(harvest_scheduler_data)

@app.route("/video_feed")
def video_feed():
    """Route to stream video from the USB camera with object detection."""
    return Response(generate_frames_with_detection(camera, interpreters, labels, last_detections), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8086)