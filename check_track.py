from ultralytics import YOLO
import cv2
import numpy as np

try:
    model = YOLO("yolov9e.pt")
    # Create a dummy frame
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Try tracking
    print("Attempting model.track()...")
    results = model.track(frame, persist=True)
    print("Tracking successful!")
    if results[0].boxes.id is not None:
        print("IDs detected (expected None for empty frame but attribute exists)")
    else:
        print("No IDs detected (expected for empty frame)")
        
except AttributeError:
    print("model.track() NOT found.")
except Exception as e:
    print(f"An error occurred: {e}")
