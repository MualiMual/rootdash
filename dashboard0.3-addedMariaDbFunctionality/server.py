from flask import Flask, render_template, jsonify, Response, g
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
import os
from dotenv import load_dotenv  # For environment variables

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# MariaDB configuration (using environment variables for security)
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'rootdash_user')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'your_password')
app.config['MYSQL_DB'] = os.getenv('MYSQL_DB', 'rootdash')

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

# Create a connection pool for MariaDB
try:
    pool = mariadb.ConnectionPool(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        database=app.config['MYSQL_DB'],
        pool_name="flask_pool",
        pool_size=5  # Adjust pool size based on your needs
    )
    print("Connection pool created successfully.")
except mariadb.Error as e:
    print(f"Error creating connection pool: {e}")
    pool = None

# Helper function to get database connection from the pool
def get_db_connection():
    """Get a database connection from the pool."""
    if pool:
        try:
            conn = pool.get_connection()
            return conn
        except mariadb.Error as e:
            print(f"Error getting connection from pool: {e}")
            return None
    else:
        print("Connection pool is not available.")
        return None

# Routes
@app.route("/")
def index():
    """Render the main dashboard page."""
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
            print(f"Error inserting sensor data: {e}")
        finally:
            conn.close()  # Return the connection to the pool

    return jsonify({**sensor_data, **system_stats})

# Other routes remain the same as in the previous code...

if __name__ == "__main__":
    # Run the app on all available network interfaces
    app.run(host="0.0.0.0", port=8086, debug=True)