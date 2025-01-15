import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

# Function to resize image without distortion by adding padding
def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    scale = min(target_size[1]/h, target_size[0]/w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Create a new image and pad the resized image to fit the target size
    padded_image = np.full((target_size[1], target_size[0], 3), 0, dtype=np.uint8)
    padded_image[(target_size[1]-new_h)//2:(target_size[1]+new_h)//2,
                 (target_size[0]-new_w)//2:(target_size[0]+new_w)//2] = resized_image
    
    return padded_image

# Try loading the Edge TPU delegate
try:
    delegate = tflite.load_delegate('libedgetpu.so.1')
    print("Edge TPU Delegate loaded successfully.")
except Exception as e:
    print(f"Failed to load Edge TPU delegate: {e}")
    delegate = None

# Load the interpreter with the TPU delegate if available, otherwise use CPU
if delegate:
    interpreter = tflite.Interpreter(
        model_path="test_data/mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite",
        experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
    )
else:
    print("Running on CPU instead of Edge TPU.")
    interpreter = tflite.Interpreter(
        model_path='test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite'
    )

interpreter.allocate_tensors()

# Load and check if image was loaded properly using OpenCV
image = cv2.imread('test_data/parrot.jpg')
if image is None:
    raise ValueError("Image not found or failed to load.")

# Resize the image with padding
image_resized = resize_with_padding(image, (224, 224))

# Prepare the image as input for the model and ensure it's cast to uint8
input_data = np.expand_dims(image_resized, axis=0).astype(np.uint8)

# Set the input tensor
input_details = interpreter.get_input_details()
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
try:
    interpreter.invoke()
except RuntimeError as e:
    print(f"Error during model inference: {e}")
    exit()

# Extract the output tensor
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Load labels and match the prediction to its label
with open("test_data/inat_bird_labels.txt", "r") as f:
    labels = f.readlines()

# Find the index of the highest confidence score
predicted_index = np.argmax(output_data)

# Print the label and confidence of the predicted class
print(f"Predicted: {labels[predicted_index].strip()}, Confidence: {output_data[0][predicted_index]}")
