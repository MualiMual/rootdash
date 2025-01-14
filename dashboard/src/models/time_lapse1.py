import cv2
import datetime
import os

def capture_time_lapse(output_folder="./media/time_lapse"):
    """
    Capture a single time-lapse image and save it to a folder with a standardized naming convention.
    :param output_folder: Folder to save the image.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame = cap.read()
    if ret:
        # Get current date and time
        now = datetime.datetime.now()
        
        # Create a subfolder for each day to organize images
        date_folder = now.strftime("%Y-%m-%d")
        daily_folder = os.path.join(output_folder, date_folder)
        os.makedirs(daily_folder, exist_ok=True)
        
        # Standardized naming convention: <experiment_id>_<timestamp>_<sequence_number>.png
        experiment_id = "exp001"  # Replace with your experiment ID
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        sequence_number = "0001"  # Since we're capturing only one image
        image_name = f"{experiment_id}_{timestamp}_{sequence_number}.png"
        image_path = os.path.join(daily_folder, image_name)
        
        # Save image with lossless compression (PNG format)
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Saved {image_path}")
    else:
        print("Error: Failed to capture image.")

    # Release the camera immediately
    cap.release()

# Example usage
# capture_time_lapse(output_folder="./media/time_lapse")