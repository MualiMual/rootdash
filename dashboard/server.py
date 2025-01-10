from flask import Flask, render_template, jsonify, Response
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import io
import base64
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
    """Fetch growth data from the database and return a base64-encoded image of the graph."""
    try:
        # Fetch data from the database
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT plant_name, height, time_after_planting FROM growth_rate ORDER BY time_after_planting")
            rows = cursor.fetchall()
            conn.close()

            # Debugging: Print fetched data
            print("Fetched Data:", rows)

            if not rows:
                return jsonify({"error": "No growth data found"}), 404

            # Organize data for plotting
            data = {}
            for row in rows:
                plant_name = row[0]
                height = float(row[1])  # Convert Decimal to float for plotting
                time_after_planting = row[2]

                if plant_name not in data:
                    data[plant_name] = {"time": [], "height": []}
                data[plant_name]["time"].append(time_after_planting)
                data[plant_name]["height"].append(height)

            # Debugging: Print organized data
            print("Organized Data:", data)

            # Create the growth graph
            plt.figure(figsize=(10, 6))  # Increase figure size for better readability
            for plant_name, values in data.items():
                plt.plot(values["time"], values["height"], label=plant_name, marker='o')  # Add markers for data points

            plt.title("Plant Growth Over Time")
            plt.xlabel("Time (Days)")
            plt.ylabel("Height (cm)")
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.grid(True)
            plt.tight_layout()  # Adjust layout to prevent overlap

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")  # Ensure the entire plot is saved
            buf.seek(0)
            plt.close()

            # Encode the image to base64
            image_base64 = base64.b64encode(buf.read()).decode("utf-8")
            print(f"Generated graph with base64 length: {len(image_base64)}")  # Debugging
            return jsonify({"image": image_base64})
        else:
            return jsonify({"error": "Failed to connect to the database"}), 500
    except Exception as e:
        print(f"Error generating growth graph: {e}")
        return jsonify({"error": "Failed to generate growth graph"}), 500

@app.route("/stored_growth_graph")
def stored_growth_graph():
    """Return the latest stored growth graph image from MariaDB."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT image FROM growth_graph ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            return jsonify({"image": row[0]})
        else:
            return jsonify({"error": "No graph images found"}), 404
    return jsonify({"error": "Failed to connect to the database"}), 500

@app.route("/growth_rate")
def growth_rate():
    """Return growth rate data from MariaDB."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT plant_name, rate, height, time_after_planting FROM growth_rate")
        data = cursor.fetchall()
        conn.close()
        return jsonify([{
            "plant_name": row[0],
            "rate": row[1],
            "height": row[2],
            "time_after_planting": row[3]
        } for row in data])
    return jsonify([])

# Function to calculate bounding box size and track growth
def track_plant_growth(detections):
    """Calculate bounding box size, track changes over time, and estimate growth rates."""
    for detection in detections:
        plant_name = detection["plant_name"]
        x_min, y_min, x_max, y_max = detection["bbox"]

        # Calculate bounding box width and height in pixels
        width_px = x_max - x_min
        height_px = y_max - y_min

        # Convert pixel measurements to real-world dimensions (e.g., cm)
        width_cm = width_px / PIXELS_PER_CM
        height_cm = height_px / PIXELS_PER_CM

        # Store the size for this plant
        plant_sizes[plant_name].append({
            "time": time.time(),  # Current timestamp
            "width_cm": width_cm,
            "height_cm": height_cm
        })

        # Calculate growth rate (if there are at least 2 data points)
        if len(plant_sizes[plant_name]) >= 2:
            prev_size = plant_sizes[plant_name][-2]
            curr_size = plant_sizes[plant_name][-1]

            time_diff = curr_size["time"] - prev_size["time"]
            width_growth_rate = (curr_size["width_cm"] - prev_size["width_cm"]) / time_diff
            height_growth_rate = (curr_size["height_cm"] - prev_size["height_cm"]) / time_diff

            print(f"Plant: {plant_name}, Width Growth Rate: {width_growth_rate:.2f} cm/s, Height Growth Rate: {height_growth_rate:.2f} cm/s")

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