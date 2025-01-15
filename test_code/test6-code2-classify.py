from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image

# Specify the TensorFlow model, labels, and image
model_file = 'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite'
label_file = 'inat_bird_labels.txt'
image_file = 'parrot.jpg'

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
#interpreter = edgetpu.make_interpreter(model_file = 'mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite')
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
#image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
image = Image.open(image_file).convert('RGB').resize(size, Image.LANCZOS)
# Run an inference
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(label_file)
for c in classes:
    print(f'{labels.get(c.id, c.id)} : {c.score:.5f}')
