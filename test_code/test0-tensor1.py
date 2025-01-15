import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="test_data/mobilenet_v2_1.0_224_inat_bird_quant.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load an image and resize it to the expected input shape
image = Image.open("test_data/parrot.jpg").resize((224, 224))
input_data = np.expand_dims(image, axis=0)

# Preprocess the image if required by the model
input_data = np.array(input_data, dtype=np.uint8)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run the inference
interpreter.invoke()

# Extract the output
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output:", output_data)
