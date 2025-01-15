import cv2

for i in range(0, 32):  # Test up to /dev/video31
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at /dev/video{i}")
        ret, frame = cap.read()
        if ret:
            print(f"Successfully captured frame from /dev/video{i}")
            cv2.imshow(f"Camera /dev/video{i}", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"Could not read frame from /dev/video{i}")
        cap.release()
    else:
        print(f"No camera at /dev/video{i}")
