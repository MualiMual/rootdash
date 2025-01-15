import tflite_runtime.interpreter as tflite

def load_models(models):
    """Load all interpreters and labels."""
    interpreters = {}
    labels = {}

    for category, paths in models.items():
        try:
            # Initialize the TensorFlow Lite interpreter with Edge TPU delegate
            interpreter = tflite.Interpreter(
                model_path=paths["model_path"],
                experimental_delegates=[tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')]
            )
            interpreter.allocate_tensors()  # Allocate memory for the model
            interpreters[category] = interpreter

            # Load the labels for the model
            with open(paths["label_path"], "r") as f:
                labels[category] = f.readlines()

            print(f"Edge TPU Delegate loaded successfully for {category} detection.")
        except Exception as e:
            print(f"Error loading {category} model or labels: {e}")

    return interpreters, labels