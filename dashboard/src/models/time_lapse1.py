import cv2
import datetime
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def capture_single_photo(output_folder="/home/boss/BASE/dev_tpu/coral/dashboard/media/time_lapse", camera_index=0, experiment_id="exp001", camera_settings=None, max_retries=3):
    """
    Capture a single photo and save it to a folder with a standardized naming convention.
    
    :param output_folder: Folder to save the image (default: "/home/boss/BASE/dev_tpu/coral/dashboard/media/time_lapse").
    :param camera_index: Index of the camera to use (default: 0).
    :param experiment_id: Unique identifier for the experiment (default: "exp001").
    :param camera_settings: Dictionary of camera settings (e.g., resolution, brightness).
    :param max_retries: Maximum number of retries if the camera is busy.
    :return: Tuple (success, message) indicating whether the capture was successful.
    """
    # Log output folder details
    logging.info(f"Output folder: {output_folder}")
    logging.info(f"Output folder exists: {os.path.exists(output_folder)}")
    logging.info(f"Output folder is writable: {os.access(output_folder, os.W_OK)}")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Check if the output folder is writable
    if not os.access(output_folder, os.W_OK):
        logging.error(f"Output folder is not writable: {output_folder}")
        return False, f"Output folder is not writable: {output_folder}"

    # Retry mechanism
    for attempt in range(max_retries):
        # Open the camera
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logging.error(f"Attempt {attempt + 1}: Could not open camera at index {camera_index}.")
            time.sleep(1)  # Wait for 1 second before retrying
            continue

        try:
            # Apply camera settings if provided
            if camera_settings:
                for setting, value in camera_settings.items():
                    if hasattr(cv2, setting):
                        cap.set(getattr(cv2, setting), value)
                        logging.info(f"Applied camera setting: {setting} = {value}")
                    else:
                        logging.warning(f"Unsupported camera setting: {setting}")

            # Capture a single frame
            ret, frame = cap.read()
            if ret:
                # Get current date and time
                now = datetime.datetime.now()

                # Create a subfolder for each day to organize images
                date_folder = now.strftime("%Y-%m-%d")
                daily_folder = os.path.join(output_folder, date_folder)
                os.makedirs(daily_folder, exist_ok=True)

                # Standardized naming convention: <experiment_id>_<timestamp>.png
                timestamp = now.strftime("%Y%m%d_%H%M%S")
                image_name = f"{experiment_id}_{timestamp}.png"
                image_path = os.path.join(daily_folder, image_name)

                # Save image with lossless compression (PNG format)
                if not cv2.imwrite(image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    logging.error(f"Error: Failed to save image to {image_path}")
                    return False, f"Failed to save image to {image_path}"
                logging.info(f"Saved {image_path}")
                return True, f"Successfully captured and saved {image_path}"
            else:
                logging.error(f"Attempt {attempt + 1}: Failed to capture image.")
                time.sleep(1)  # Wait for 1 second before retrying
        except Exception as e:
            logging.error(f"Attempt {attempt + 1}: Error during photo capture: {e}")
            time.sleep(1)  # Wait for 1 second before retrying
        finally:
            # Release the camera
            cap.release()
            logging.info("Camera released.")

    logging.error(f"Failed to capture image after {max_retries} attempts.")
    return False, "Failed to capture image after multiple attempts."

# Example usage
if __name__ == "__main__":
    # Example camera settings (optional)
    camera_settings = {
        "CAP_PROP_FRAME_WIDTH": 1920,  # Set resolution width
        "CAP_PROP_FRAME_HEIGHT": 1080,  # Set resolution height
        "CAP_PROP_BRIGHTNESS": 0.5,     # Set brightness
    }

    # Capture a single photo
    success, message = capture_single_photo(
        output_folder="/home/boss/BASE/dev_tpu/coral/dashboard/media/time_lapse",  # Correct full path
        camera_index=0,
        experiment_id="exp001",
        camera_settings=camera_settings  # Optional: Pass camera settings
    )

    if success:
        print(message)
    else:
        print(f"Error: {message}")