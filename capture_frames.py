import cv2
import json
import os
import time
import requests
import numpy as np

def capture_frames():
    # Load target cameras
    with open('d:/FYP/edgeagent/target_cameras.json', 'r') as f:
        cameras = json.load(f)
        
    output_dir = 'd:/FYP/edgeagent/roi_labeler/images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Starting capture for {len(cameras)} cameras...")
    
    for i, cam in enumerate(cameras):
        name = cam.get('cameraName')
        if not name:
            name = f"camera_{i}"
            
        # Sanitize filename
        safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip()
        safe_name = safe_name.replace(' ', '_')
        
        filename = os.path.join(output_dir, f"{safe_name}.jpg")
        
        if os.path.exists(filename):
            print(f"Skipping {name}, already exists.")
            continue
            
        # Prefer static image URL for reference frame
        url = cam.get('url')
        if not url:
            url = cam.get('streamUrl')
            
        if not url:
            print(f"No URL for {name}")
            continue
            
        print(f"Downloading {name} from {url}...")
        
        try:
            # Try downloading as image first
            time.sleep(2) # Wait 2 seconds before request
            response = requests.get(url, timeout=30) # Increased timeout
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Saved {filename}")
                
                # Verify it's a valid image
                img = cv2.imread(filename)
                if img is None:
                    print(f"Invalid image downloaded for {name}, deleting.")
                    os.remove(filename)
                else:
                    continue # Success
        except Exception as e:
            print(f"Error downloading image: {e}")

        # Fallback to VideoCapture if direct download failed or wasn't an image
        print(f"Falling back to stream capture for {name}...")
        cap = cv2.VideoCapture(url)
        success = False
        for _ in range(30):
            ret, frame = cap.read()
            if ret:
                cv2.imwrite(filename, frame)
                print(f"Saved {filename} from stream")
                success = True
                break
            time.sleep(0.1)
        cap.release()
            
        if not success:
            print(f"Failed to capture {name}")
        
    print("Capture complete.")

if __name__ == "__main__":
    capture_frames()
