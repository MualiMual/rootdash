from flask import Flask, render_template, jsonify, Response
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import io
import os
import base64
import mariadb
from datetime import datetime, timedelta
from src.utils.camera import get_camera
from src.utils.edgedevice import load_models  # Import from edge device
# Models----------------
from src.models.time_lapse1 import capture_time_lapse
from src.models.image_analysis2 import analyze_image
from src.models.object_detection import generate_frames as generate_frames_with_detection  # Import object detection
import threading

app = Flask(__name__)

# MariaDB configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'rootdash_user'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'rootdash'

# Time-lapse configuration
app.config['TIME_LAPSE_FOLDER'] = "./media/time_lapse"  # Folder to save time-lapse images

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

@app.route("/start_time_lapse", methods=["POST"])
def start_time_lapse():
    """Start capturing time-lapse images in a separate thread."""
    try:
        # Start the time-lapse capture in a separate thread
        thread = threading.Thread(target=capture_time_lapse, kwargs={
            'output_folder': app.config['TIME_LAPSE_FOLDER'],
            'interval': 3600,  # 1 hour interval
            'num_images': 10   # Capture 10 images
        })
        thread.daemon = True  # Daemonize thread to exit when the main program exits
        thread.start()
        return jsonify({"message": "Time-lapse capture started successfully!"}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to start time-lapse capture: {str(e)}"}), 500

@app.route("/analyze_images", methods=["POST"])
def analyze_images():
    """Analyze all time-lapse images."""
    try:
        image_folder = app.config['TIME_LAPSE_FOLDER']
        if not os.path.exists(image_folder):
            return jsonify({"message": "No time-lapse images found!"}), 404

        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            if os.path.isfile(image_path):
                plant_data = analyze_image(image_path)
                for data in plant_data:
                    # Insert data into the database
                    insert_time_lapse_data(image_name, data["width"], data["height"])
        return jsonify({"message": "Image analysis completed successfully!"}), 200
    except Exception as e:
        return jsonify({"message": f"Failed to analyze images: {str(e)}"}), 500

def insert_time_lapse_data(image_name, width, height):
    """Insert time-lapse data into the database."""
    timestamp = image_name.split("_")[1].split(".")[0]  # Extract timestamp from filename
    query = """
        INSERT INTO time_lapse_data (timestamp, plant_id, width, height, image_path)
        VALUES (%s, %s, %s, %s, %s)
    """
    values = (timestamp, 1, width, height, image_name)  # Replace 1 with actual plant ID
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        conn.close()

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
            cursor.execute('''
                SELECT 
                    p.name AS plant_name, 
                    gr.height, 
                    gr.time_after_planting
                FROM 
                    growth_rate gr
                JOIN 
                    plants p ON gr.plant_id = p.id
                ORDER BY 
                    gr.time_after_planting
            ''')
            rows = cursor.fetchall()
            conn.close()

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
        cursor.execute('''
            SELECT 
                gr.rate, 
                gr.height, 
                gr.time_after_planting, 
                p.name AS plant_name
            FROM 
                growth_rate gr
            JOIN 
                plants p ON gr.plant_id = p.id
        ''')
        data = cursor.fetchall()
        conn.close()
        return jsonify([{
            "rate": row[0],
            "height": row[1],
            "time_after_planting": row[2],
            "plant_name": row[3]
        } for row in data])
    return jsonify([])

@app.route("/seasonal_status")
def seasonal_status():
    """Calculate seasonal status based on growth_rate data."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                gr.time_after_planting, 
                p.name AS plant_name
            FROM 
                growth_rate gr
            JOIN 
                plants p ON gr.plant_id = p.id
        ''')
        data = cursor.fetchall()
        conn.close()

        seasonal_status_data = []
        for row in data:
            time_after_planting = row[0]
            plant_name = row[1]

            # Calculate start_date dynamically if not available in the database
            start_date = (datetime.now() - timedelta(days=time_after_planting)).strftime("%Y-%m-%d")

            # Calculate current_stage based on time_after_planting
            if time_after_planting < 30:
                current_stage = "Early Growth"
            elif 30 <= time_after_planting < 60:
                current_stage = "Mid Growth"
            else:
                current_stage = "Late Growth"

            seasonal_status_data.append({
                "plant_name": plant_name,
                "start_date": start_date,  # Include start_date in the response
                "current_stage": current_stage
            })

        return jsonify(seasonal_status_data)
    return jsonify([])

@app.route("/harvest_scheduler")
def harvest_scheduler():
    """Predict harvest dates based on growth_rate data."""
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                gr.time_after_planting, 
                p.name AS plant_name
            FROM 
                growth_rate gr
            JOIN 
                plants p ON gr.plant_id = p.id
        ''')
        data = cursor.fetchall()
        conn.close()

        harvest_scheduler_data = []
        for row in data:
            time_after_planting = row[0]
            plant_name = row[1]

            # Predict harvest date (assuming a fixed growth period of 90 days)
            planting_date = datetime.now() - timedelta(days=time_after_planting)
            predicted_harvest_date = planting_date + timedelta(days=90)

            harvest_scheduler_data.append({
                "plant_name": plant_name,
                "predicted_harvest_date": predicted_harvest_date.strftime("%Y-%m-%d")
            })

        return jsonify(harvest_scheduler_data)
    return jsonify([])

@app.route("/video_feed")
def video_feed():
    """Route to stream video from the USB camera with object detection."""
    return Response(generate_frames_with_detection(camera, interpreters, labels, last_detections), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8086)