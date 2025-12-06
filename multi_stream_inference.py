import cv2
import json
import time
import threading
import queue
import os
import yt_dlp
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from database_manager import DatabaseManager

# Constants
TARGET_FPS = 5.0
FRAME_INTERVAL = 1.0 / TARGET_FPS
ROI_FILE = "d:/FYP/edgeagent/roi_labeler/road_rois.json"
CAMERAS_FILE = "d:/FYP/edgeagent/target_cameras.json"

# Load ROIs
try:
    with open(ROI_FILE, "r") as f:
        ROIS = json.load(f)
    print(f"Loaded ROIs for {len(ROIS)} cameras.")
    # Convert to Shapely polygons
    ROIS_SHAPELY = {}
    for cam_name, rois in ROIS.items():
        ROIS_SHAPELY[cam_name] = {}
        for roi_name, points in rois.items():
            if len(points) >= 3:
                ROIS_SHAPELY[cam_name][roi_name] = Polygon(points)
except Exception as e:
    print(f"Error loading ROIs: {e}")
    ROIS_SHAPELY = {}

def get_stream_url(url):
    """Resolve YouTube URL or format .stream URL properly."""
    if "youtube.com" in url or "youtu.be" in url:
        ydl_opts = {"format": "best[ext=mp4]", "quiet": True, "noplaylist": True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info["url"]
        except:
            return None
    
    # Handle Wowza/Streamlock .stream extensions
    if url.endswith(".stream") or url.endswith(".stream/"):
        if not url.endswith("/"):
            url += "/"
        return url + "playlist.m3u8"
        
    return url

class CameraStream:
    def __init__(self, name, url):
        self.name = name
        self.base_url = url
        self.resolved_url = None
        self.cap = None
        self.q = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
        self.last_inference_time = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        # Initial connection attempt
        self._connect()
        
        while self.running:
            if not self.cap or not self.cap.isOpened():
                time.sleep(5)
                self._connect()
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                # Stream disconnected
                self.cap.release()
                self.cap = None
                time.sleep(1) 
                continue
            
            # Keep only the latest frame
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
            
    def _connect(self):
        """Attempts to resolve URL and connect"""
        try:
            self.resolved_url = get_stream_url(self.base_url)
            if self.resolved_url:
                # print(f"Connecting to {self.name} at {self.resolved_url}...")
                self.cap = cv2.VideoCapture(self.resolved_url)
                if self.cap.isOpened():
                    # print(f"Connected to {self.name}")
                    pass
                else:
                    # print(f"Failed to open {self.name}")
                    pass
        except Exception as e:
            print(f"Connection error for {self.name}: {e}")

    def read(self):
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()

def main():
    # Load cameras
    with open(CAMERAS_FILE, "r") as f:
        cameras_config = json.load(f)

    # Initialize DB
    db = DatabaseManager()

    # Initialize Model
    try:
        model = YOLO("yolov9e.pt")
        target_classes = ["car", "truck", "motorcycle"]
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Initialize Streams
    streams = []
    print(f"Initializing {len(cameras_config)} streams...")
    
    for i, cam in enumerate(cameras_config):
        # Sanitize name
        raw_name = cam.get('cameraName', f'camera_{i}')
        safe_name = "".join([c for c in raw_name if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).strip()
        safe_name = safe_name.replace(' ', '_')
        
        url = cam.get('streamUrl') or cam.get('url')
        if url:
            s = CameraStream(safe_name, url)
            streams.append(s)
            s.start() # Start immediately, non-blocking

    print("Streams started. Starting inference loop...")
    
    print_timer = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            processed_count = 0
            
            for stream in streams:
                if current_time - stream.last_inference_time < FRAME_INTERVAL:
                    continue

                frame = stream.read()
                if frame is None:
                    continue

                stream.last_inference_time = current_time
                processed_count += 1
                
                # Inference
                results = model(frame, verbose=False)
                
                vehicle_counts = {}
                congestion_levels = {}
                
                # Get ROIs for this camera
                cam_rois = ROIS_SHAPELY.get(stream.name, {})
                
                # Initialize counts
                for roi_name in cam_rois:
                    vehicle_counts[roi_name] = 0
                
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        cls = int(box.cls)
                        if model.names[cls] in target_classes:
                            # Get center point
                            xyxy = box.xyxy[0].tolist()
                            cx = (xyxy[0] + xyxy[2]) / 2
                            cy = (xyxy[1] + xyxy[3]) / 2
                            pt = Point(cx, cy)
                            
                            for roi_name, poly in cam_rois.items():
                                if pt.within(poly):
                                    vehicle_counts[roi_name] += 1
                                    break 
                
                # Calculate congestion
                total_vehicles = 0
                for roi_name, count in vehicle_counts.items():
                    total_vehicles += count
                    if count > 10: level = "High"
                    elif count > 5: level = "Medium"
                    else: level = "Low"
                    congestion_levels[roi_name] = level

                # Log to DB
                if vehicle_counts: 
                    db.insert_frame_metrics(stream.name, current_time, vehicle_counts, congestion_levels)
                    
            if current_time - print_timer > 5.0:
                print(f"Running... PROCESSED FRAMES: {processed_count}")
                print_timer = current_time

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        for s in streams:
            s.stop()
        db.close()

if __name__ == "__main__":
    main()
