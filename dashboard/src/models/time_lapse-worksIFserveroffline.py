import cv2
import datetime
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_camera_available(camera_index=0):
    """Check if the camera is available."""
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.release()
        return True
    return False

def capture_single_photo(output_folder="/home/boss/BASE/dev_tpu/coral/dashboard/media/time_lapse", camera_index=0, experiment_id="exp001", camera_settings=None):
    """
    Capture a single photo and save it to a folder with a standardized naming convention.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        logging.info(f"Created output folder: {output_folder}")

    if not is_camera_available(camera_index):
        logging.error("Camera is currently in use or unavailable.")
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logging.error("Failed to open the camera.")
        return

    # Apply camera settings
    if camera_settings:
        for setting, value in camera_settings.items():
            cap.set(getattr(cv2, setting), value)
            logging.info(f"Applied camera setting: {setting} = {value}")

    ret, frame = cap.read()
    if ret:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_id}_{timestamp}.png"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        logging.info(f"Saved {filepath}")
    else:
        logging.error("Failed to capture image.")

    cap.release()
    logging.info("Camera released.")

# Example usage
if __name__ == "__main__":
    camera_settings = {
        "CAP_PROP_FRAME_WIDTH": 1920,
        "CAP_PROP_FRAME_HEIGHT": 1080,
        "CAP_PROP_BRIGHTNESS": 0.5,
    }
    capture_single_photo(camera_settings=camera_settings)
