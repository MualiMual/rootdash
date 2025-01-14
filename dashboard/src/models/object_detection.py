import cv2
import numpy as np

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

def generate_frames(camera, interpreters, labels, last_detections):
    """Generate video frames with object detection."""
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