import time
import board
import busio
import adafruit_ads1x15.ads1015 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import adafruit_tcs34725
import adafruit_icm20x
import adafruit_lps2x
import adafruit_shtc3
import cv2
import psutil
import socket
import tflite_runtime.interpreter as tflite
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import io
import base64
import random

app = Flask(__name__)

# Global variable to store the last 5 inference results
last_detections = deque(maxlen=5)

# Initialize I2C
try:
    i2c = busio.I2C(board.SCL, board.SDA)
    print("I2C initialized successfully!")
except Exception as e:
    print(f"Error initializing I2C: {e}")
    i2c = None

# Initialize ADS1015
try:
    if i2c:
        ads = ADS.ADS1015(i2c)
        print("ADS1015 initialized successfully!")
    else:
        ads = None
        print("ADS1015 not initialized due to I2C error.")
except Exception as e:
    print(f"Error initializing ADS1015: {e}")
    ads = None

# Define analog input channel (A0)
channel = AnalogIn(ads, ADS.P0) if ads else None

# Initialize TCS34725 sensor
try:
    if i2c:
        color_sensor = adafruit_tcs34725.TCS34725(i2c)
        print("TCS34725 sensor initialized successfully!")
    else:
        color_sensor = None
        print("TCS34725 not initialized due to I2C error.")
except Exception as e:
    print(f"Error initializing TCS34725: {e}")
    color_sensor = None

# Initialize ICM-20948 sensor
try:
    if i2c:
        motion_sensor = adafruit_icm20x.ICM20948(i2c, address=0x68)
        print("ICM-20948 sensor initialized successfully!")
    else:
        motion_sensor = None
        print("ICM-20948 not initialized due to I2C error.")
except Exception as e:
    print(f"Error initializing ICM-20948: {e}")
    motion_sensor = None

# Initialize LPS22HB sensor
try:
    if i2c:
        pressure_sensor = adafruit_lps2x.LPS22(i2c, address=0x5C)
        print("LPS22HB sensor initialized successfully!")
    else:
        pressure_sensor = None
        print("LPS22HB not initialized due to I2C error.")
except Exception as e:
    print(f"Error initializing LPS22HB: {e}")
    pressure_sensor = None

# Initialize SHTC3 sensor
try:
    if i2c:
        sht_sensor = adafruit_shtc3.SHTC3(i2c)
        print("SHTC3 sensor initialized successfully!")
    else:
        sht_sensor = None
        print("SHTC3 not initialized due to I2C error.")
except Exception as e:
    print(f"Error initializing SHTC3: {e}")
    sht_sensor = None

# Initialize USB Camera
def get_camera():
    """Try to initialize the USB camera."""
    try:
        camera = cv2.VideoCapture(0)  # Use the default camera (index 0)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            return None
        print("USB Camera initialized successfully!")
        return camera
    except Exception as e:
        print(f"Error initializing USB Camera: {e}")
        return None

camera = get_camera()

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
            labels[category] = f.readlines()

        print(f"Edge TPU Delegate loaded successfully for {category} detection.")
    except Exception as e:
        print(f"Error loading {category} model or labels: {e}")

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

def generate_frames():
    """Generate video frames with object detection."""
    global last_detections
    while True:
        if camera is None:
            break
        success, frame = camera.read()
        if not success:
            break

        # Apply background subtraction to detect motion
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around moving objects
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter out small contours
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Process frame with each model
        for category, interpreter in interpreters.items():
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

            # Get the label
            label = labels[category][predicted_label].strip()

            # Add the result to the last_detections list
            last_detections.append({
                "category": category,
                "label": label,
                "confidence": float(confidence)
            })

            # Display the result on the frame
            cv2.putText(frame, f"{category.upper()}: {label} ({confidence:.2f})",
                        (10, 30 + 40 * list(interpreters.keys()).index(category)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Conversion Functions
def celsius_to_fahrenheit(celsius):
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def hpa_to_inhg(hpa):
    """Convert hPa to inHg (inches of mercury)."""
    return hpa * 0.02953

# System Monitoring Functions
def get_cpu_usage():
    """Get CPU usage as a percentage."""
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    """Get RAM usage as a percentage."""
    return psutil.virtual_memory().percent

def get_storage_usage():
    """Get storage usage as a percentage."""
    return psutil.disk_usage('/').percent

def get_network_status():
    """Get network connection status and IP address."""
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return {
            "connected": True,
            "ip_address": ip_address
        }
    except Exception:
        return {
            "connected": False,
            "ip_address": "N/A"
        }

# Growth Rate Data
growth_data = {
    "Plant A": [10, 15, 20, 25, 30],
    "Plant B": [5, 10, 15, 20, 25],
    "Plant C": [8, 12, 16, 20, 24]
}

def generate_growth_graph():
    """Generate a growth graph using Matplotlib."""
    plt.figure(figsize=(6, 4))
    for plant, rates in growth_data.items():
        plt.plot(rates, label=plant)
    plt.title('Plant Growth Over Time')
    plt.xlabel('Time (Days)')
    plt.ylabel('Height (cm)')
    plt.legend()
    plt.grid(True)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the image to base64
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

# Seasonal Status Data
seasonal_status_data = [
    {"plant_name": "Tomato", "start_date": "2023-09-01", "harvest_date": "2023-11-15", "current_stage": random.randint(30, 70)},
    {"plant_name": "Lettuce", "start_date": "2023-09-10", "harvest_date": "2023-11-01", "current_stage": random.randint(40, 80)},
    {"plant_name": "Pepper", "start_date": "2023-08-20", "harvest_date": "2023-11-10", "current_stage": random.randint(50, 90)}
]

# Harvest Scheduler Data
harvest_scheduler_data = [
    {"plant_name": "Tomato", "predicted_harvest_date": "2023-11-15"},
    {"plant_name": "Lettuce", "predicted_harvest_date": "2023-11-01"},
    {"plant_name": "Pepper", "predicted_harvest_date": "2023-11-10"}
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sensor_data')
def sensor_data():
    try:
        # Read system monitoring data (always available)
        cpu_usage = get_cpu_usage()
        ram_usage = get_ram_usage()
        storage_usage = get_storage_usage()
        network_status = get_network_status()

        # Read analog sensor data (if available)
        analog_value = channel.value if channel else None
        analog_voltage = channel.voltage if channel else None

        # Read color sensor data (if available)
        if color_sensor:
            r, g, b, c = color_sensor.color_raw
        else:
            r, g, b, c = None, None, None, None

        # Read motion sensor data (if available)
        if motion_sensor:
            accel = motion_sensor.acceleration  # (x, y, z) in m/s^2
            gyro = motion_sensor.gyro           # (x, y, z) in degrees/s
            mag = motion_sensor.magnetic        # (x, y, z) in µT
        else:
            accel = (None, None, None)
            gyro = (None, None, None)
            mag = (None, None, None)

        # Read pressure and temperature data (if available)
        if pressure_sensor:
            pressure_hpa = pressure_sensor.pressure  # Pressure in hPa
            pressure_inhg = hpa_to_inhg(pressure_hpa)  # Convert to inHg
            temperature_c = pressure_sensor.temperature  # Temperature in °C
            temperature_f = celsius_to_fahrenheit(temperature_c)  # Convert to °F
        else:
            pressure_inhg = None
            temperature_f = None

        # Read SHTC3 temperature and humidity data (if available)
        if sht_sensor:
            temperature_sht_c, humidity = sht_sensor.measurements  # Temperature in °C, Humidity in %
            temperature_sht_f = celsius_to_fahrenheit(temperature_sht_c)  # Convert to °F
        else:
            temperature_sht_f = None
            humidity = None

        # Return sensor data as JSON
        return jsonify({
            "analog_value": analog_value,
            "analog_voltage": analog_voltage,
            "color_red": r,
            "color_green": g,
            "color_blue": b,
            "color_clear": c,
            "accel_x": accel[0],
            "accel_y": accel[1],
            "accel_z": accel[2],
            "gyro_x": gyro[0],
            "gyro_y": gyro[1],
            "gyro_z": gyro[2],
            "mag_x": mag[0],
            "mag_y": mag[1],
            "mag_z": mag[2],
            "pressure": pressure_inhg,
            "temperature_pressure": temperature_f,
            "temperature_sht": temperature_sht_f,
            "humidity": humidity,
            "cpu_usage": cpu_usage,
            "ram_usage": ram_usage,
            "storage_usage": storage_usage,
            "network_connected": network_status["connected"],
            "ip_address": network_status["ip_address"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/inference_data')
def inference_data():
    """Return the last 5 inference results."""
    return jsonify(list(last_detections))

@app.route('/growth_graph')
def growth_graph():
    """Return the growth graph as a base64 encoded image."""
    image_base64 = generate_growth_graph()
    return jsonify({"image": image_base64})

@app.route('/seasonal_status')
def seasonal_status():
    """Return seasonal status data."""
    return jsonify(seasonal_status_data)

@app.route('/harvest_scheduler')
def harvest_scheduler():
    """Return harvest scheduler data."""
    return jsonify(harvest_scheduler_data)

@app.route('/video_feed')
def video_feed():
    """Route to stream video from the USB camera with object detection."""
    if camera is None:
        return "Camera not connected", 404
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8086)