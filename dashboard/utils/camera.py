import cv2
import numpy as np

def get_camera():
    """Initialize the USB camera and configure autofocus settings."""
    try:
        # Initialize the camera
        camera = cv2.VideoCapture(0)  # Use the default camera (index 0)
        if not camera.isOpened():
            print("Error: Could not open camera.")
            return None

        # Configure camera properties (if supported)
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Disable autofocus (0 = off, 1 = on)
        camera.set(cv2.CAP_PROP_FOCUS, 0)  # Set initial focus value (0 = minimum focus)

        print("USB Camera initialized successfully!")
        return camera
    except Exception as e:
        print(f"Error initializing USB Camera: {e}")
        return None

def calculate_sharpness(image):
    """Calculate the sharpness of an image using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def autofocus(camera, max_iterations=10, focus_step=5):
    """
    Perform autofocus by adjusting the camera's focus setting to maximize image sharpness.

    Args:
        camera: The camera object (cv2.VideoCapture).
        max_iterations: Maximum number of focus adjustments.
        focus_step: Step size for focus adjustments.

    Returns:
        best_focus: The focus value that produced the sharpest image.
        best_sharpness: The sharpness value of the best-focused image.
    """
    best_focus = 0
    best_sharpness = 0

    # Iterate through focus values
    for focus in range(0, 255, focus_step):
        # Set focus value
        camera.set(cv2.CAP_PROP_FOCUS, focus)

        # Capture a frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Calculate sharpness
        sharpness = calculate_sharpness(frame)
        print(f"Focus: {focus}, Sharpness: {sharpness:.2f}")

        # Update best focus
        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_focus = focus

        # Display the frame (optional)
        cv2.imshow("Autofocus", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit early
            break

    # Set the best focus value
    camera.set(cv2.CAP_PROP_FOCUS, best_focus)
    print(f"Best Focus: {best_focus}, Best Sharpness: {best_sharpness:.2f}")

    return best_focus, best_sharpness