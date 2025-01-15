import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

interpreter = tflite.Interpreter(model_path='test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite',
                                  experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

image = cv2.imread('bird.jpg')
image_resized = cv2.resize(image, (224, 224))

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.expand_dims(image_resized, axis=0)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print('Predicted:', output_data)
