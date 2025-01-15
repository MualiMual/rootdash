from flask import Flask, render_template, jsonify, Response
from collections import deque
import random
import time
import matplotlib.pyplot as plt
import io
import os
import subprocess
import base64
import mariadb
from datetime import datetime, timedelta
from src.utils.camera import get_camera
from src.utils.edgedevice import load_models  # Import from edge device
# Models----------------
from src.models.time_lapse import capture_single_photo
from src.models.analyze_image import analyze_image
from src.models.object_detection import generate_frames as generate_frames_with_detection  # Import object detection
import threading
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# MariaDB configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'rootdash_user'
app.config['MYSQL_PASSWORD'] = 'your_password'
app.config['MYSQL_DB'] = 'rootdash'

# Time-lapse configuration
app.config['TIME_LAPSE_FOLDER'] = os.path.expanduser("~/BASE/dev_tpu/coral/dashboard/media/time_lapse")

# Ensure the time-lapse folder exists
os.makedirs(app.config['TIME_LAPSE_FOLDER'], exist_ok=True)

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
        logging.error(f"Error connecting to MariaDB: {e}")
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
        try:
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
        except mariadb.Error as e:
            logging.error(f"Error inserting sensor data: {e}")
        finally:
            conn.close()

    return jsonify({**sensor_data, **system_stats})
def capture_single_photo(output_folder):
    """
    Simulate capturing a single photo.
    Replace this with your actual photo capture logic.
    """
    try:
        # Simulate photo capture
        time.sleep(1)  # Simulate delay
        logging.info(f"Photo captured and saved to {output_folder}")
        return True, "Photo captured successfully"
    except Exception as e:
        logging.error(f"Error capturing photo: {e}")
        return False, str(e)

def capture_time_lapse(output_folder, interval, num_images):
    """
    Capture time-lapse images and save them to the specified folder.
    
    :param output_folder: Folder to save the images.
    :param interval: Time interval between captures (in seconds).
    :param num_images: Number of images to capture.
    """
    for i in range(num_images):
        success, message = capture_single_photo(output_folder=output_folder)
        if success:
            logging.info(f"Captured image {i + 1}/{num_images}: {message}")
        else:
            logging.error(f"Failed to capture image {i + 1}/{num_images}: {message}")
        time.sleep(interval)

@app.route("/pause_feed", methods=["POST"])
def pause_feed():
    """Pause the live feed to allow the camera to be used by other processes."""
    global is_feed_paused
    is_feed_paused = True
    logging.info("Live feed paused.")
    return jsonify({"message": "Live feed paused."}), 200

@app.route("/resume_feed", methods=["POST"])
def resume_feed():
    """Resume the live feed after the camera is released by other processes."""
    global is_feed_paused
    is_feed_paused = False
    logging.info("Live feed resumed.")

@app.route("/start_time_lapse", methods=["POST"])  # Use the @app.route decorator
def start_time_lapse():
    """Start capturing time-lapse images in a separate thread."""
    try:
        # Start the time-lapse capture in a separate thread
        thread = threading.Thread(target=capture_time_lapse, kwargs={
            'output_folder': app.config['TIME_LAPSE_FOLDER'],
            'interval': 3600,  # 1 hour interval
            'num_images': 1    # Capture 1 image
        })
        thread.daemon = True  # Daemonize thread to exit when the main program exits
        thread.start()
        return jsonify({"message": "Time-lapse capture started successfully!"}), 200
    except Exception as e:
        logging.error(f"Failed to start time-lapse capture: {e}")
        return jsonify({"message": f"Failed to start time-lapse capture: {str(e)}"}), 500

@app.route("/analyze_images", methods=["POST"])
def analyze_images():
    """Run the analyze_image.py script as a subprocess."""
    try:
        # Get the absolute path to the analyze_image.py script
        script_path = os.path.abspath("src/models/analyze_image.py")

        # Check if the script exists
        if not os.path.exists(script_path):
            logging.error(f"Script not found: {script_path}")
            return jsonify({"message": "Script not found!"}), 404

        # Run the script using subprocess
        logging.info(f"Running script: {script_path}")
        result = subprocess.run(
            ["python", script_path],  # Command to execute the script
            capture_output=True,      # Capture stdout and stderr
            text=True                 # Return output as a string
        )

        # Log the script's output and errors
        if result.returncode == 0:
            logging.info(f"Script output: {result.stdout}")
            return jsonify({"message": "Script executed successfully!", "output": result.stdout}), 200
        else:
            logging.error(f"Script failed with error: {result.stderr}")
            return jsonify({"message": "Script execution failed!", "error": result.stderr}), 500

    except Exception as e:
        logging.error(f"Error running script: {e}")
        return jsonify({"message": f"Failed to run script: {str(e)}"}), 500

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
        logging.error(f"Error generating growth graph: {e}")
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