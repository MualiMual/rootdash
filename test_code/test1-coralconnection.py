import tflite_runtime.interpreter as tflite
import platform

def load_delegate():
    try:
        machine = platform.machine()
        if machine == 'aarch64':
            print("Loading Edge TPU Delegate for ARM 64-bit...")
            delegate = tflite.load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')
            print("Edge TPU Delegate loaded successfully.")
        elif machine == 'x86_64':
            print("Loading Edge TPU Delegate for x86_64...")
            delegate = tflite.load_delegate('/usr/lib/x86_64-linux-gnu/libedgetpu.so.1')
            print("Edge TPU Delegate loaded successfully.")
        else:
            print(f"This platform ({machine}) is not supported for Edge TPU.")
    except ValueError as e:
        print(f"Failed to load Edge TPU delegate: {e}")

if __name__ == '__main__':
    load_delegate()
