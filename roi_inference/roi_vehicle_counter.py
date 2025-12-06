import cv2
import torch
import numpy as np
import yt_dlp
from ultralytics import YOLO
import json
import os
from shapely.geometry import Point, Polygon
import time
import math  # For polygon area calculation
try:
    from accident_detector import AccidentDetector
except ImportError:
    print("Warning: AccidentDetector module not found. Running without accident detection.")
    class AccidentDetector:
        def update(self, tracks, timestamp):
            return []

# --- Congestion Control Constants ---
LANE_CONGESTION_THRESHOLDS = {
    "Low": 0.00005,
    "Medium": 0.0001,
}

GREEN_TIMES = {
    "Low": 10,
    "Medium": 20,
    "High": 35,
}

LANE_COLORS = {
    "Low": (0, 255, 0),
    "Medium": (0, 255, 255),
    "High": (0, 0, 255),
    "Default": (255, 255, 255),
}
# --- End Congestion Control Constants ---


def get_stream_url(youtube_url):
    """Extract direct stream URL from a YouTube video using yt-dlp."""
    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=False)
            return info_dict["url"]
    except Exception as e:
        print(f"Error getting stream URL from YouTube: {e}")
        return None


def main(stream_url, roi_file_path):
    """Perform real-time vehicle detection and ROI congestion estimation."""
    # Load YOLOv9e model
    try:
        model = YOLO("yolov9e.pt")
    except Exception as e:
        print("Error loading YOLOv9e model.")
        print(e)
        return

    # Load ROIs
    rois_shapely = {}
    rois_display = {}
    rois_areas = {}
    try:
        with open(roi_file_path, "r") as f:
            loaded_rois = json.load(f)
            for name, points_list in loaded_rois.items():
                if len(points_list) >= 3:
                    polygon = Polygon(points_list)
                    rois_shapely[name] = polygon
                    rois_display[name] = np.array(points_list, np.int32).reshape(
                        (-1, 1, 2)
                    )
                    rois_areas[name] = polygon.area
                else:
                    print(f"Skipping ROI '{name}': Not enough points.")
        if not rois_shapely:
            print("No valid ROIs loaded from the JSON file.")
            return
        print(f"Loaded {len(rois_shapely)} ROIs from {roi_file_path}")
    except Exception as e:
        print(f"Error loading ROI file: {e}")
        return

    # Video Capture setup
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video opened: {frame_width}x{frame_height} @ {fps} FPS")

    target_classes = ["car", "truck", "motorcycle"]

    cv2.namedWindow("ROI Vehicle Counter", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ROI Vehicle Counter", 1280, 720)
    print("\nStarting inference. Press 'q' to quit.")

    lane_stats = {
        name: {"count": 0, "density": 0.0, "congestion": "Low"}
        for name in rois_shapely.keys()
    }

    def get_congestion_level(count, area):
        if area == 0:
            return "Low"
        density = count / area
        if density < LANE_CONGESTION_THRESHOLDS["Low"]:
            return "Low"
        elif density < LANE_CONGESTION_THRESHOLDS["Medium"]:
            return "Medium"
        else:
            return "High"

    def select_next_lane():
        high = [
            lane for lane, stats in lane_stats.items() if stats["congestion"] == "High"
        ]
        if high:
            return high[0]
        medium = [
            lane
            for lane, stats in lane_stats.items()
            if stats["congestion"] == "Medium"
        ]
        if medium:
            return medium[0]
        low = [
            lane for lane, stats in lane_stats.items() if stats["congestion"] == "Low"
        ]
        return low[0] if low else None

    current_green_lane = None
    last_light_change_time = time.time()
    MIN_RED_TIME = 5

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or error reading frame.")
                break

            # Initialize Accident Detector
            if 'detector' not in locals():
                detector = AccidentDetector()

            # Run tracking instead of just detection
            results = model.track(frame, persist=True, verbose=False)
            roi_vehicle_counts = {name: 0 for name in rois_shapely.keys()}
            
            current_tracks = []

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    cls = int(box.cls)
                    detected_class_name = model.names[cls]
                    
                    if detected_class_name in target_classes:
                        # Extract track info
                        track_id = int(box.id) if box.id is not None else -1
                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Add to current tracks for accident detection
                        class TrackObj:
                            def __init__(self, tid, bbox, c):
                                self.id = tid
                                self.box = bbox
                                self.cls = c
                        
                        current_tracks.append(TrackObj(track_id, [x1, y1, x2, y2], detected_class_name))

                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        vehicle_point = Point(center_x, center_y)
                        
                        # ROI Counting
                        for roi_name, roi_polygon in rois_shapely.items():
                            if vehicle_point.within(roi_polygon):
                                roi_vehicle_counts[roi_name] += 1
                                # Draw bounding box with ID
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                                break
            
            # Run Accident Detection
            accidents = detector.update(current_tracks, time.time())
            
            # Draw Accidents
            for acc in accidents:
                abox = acc['box']
                cv2.rectangle(frame, (int(abox[0]), int(abox[1])), (int(abox[2]), int(abox[3])), (0, 0, 255), 3)
                cv2.putText(frame, f"ALERT: {acc['type']}", (int(abox[0]), int(abox[1]) - 25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Update stats
            for roi_name, count in roi_vehicle_counts.items():
                area = rois_areas[roi_name]
                lane_stats[roi_name]["count"] = count
                lane_stats[roi_name]["density"] = count / area if area > 0 else 0.0
                lane_stats[roi_name]["congestion"] = get_congestion_level(count, area)

            # Traffic light simulation
            if time.time() - last_light_change_time > MIN_RED_TIME:
                next_lane = select_next_lane()
                if next_lane and next_lane != current_green_lane:
                    current_green_lane = next_lane
                    last_light_change_time = time.time()
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Green for: {current_green_lane} ({lane_stats[current_green_lane]['congestion']})"
                    )
            elif (
                current_green_lane
                and (time.time() - last_light_change_time)
                > GREEN_TIMES[lane_stats[current_green_lane]["congestion"]]
            ):
                print(f"[{time.strftime('%H:%M:%S')}] Red for: {current_green_lane}")
                current_green_lane = None
                last_light_change_time = time.time()

            # --- Draw top-left info box ---
            overlay = frame.copy()
            cv2.rectangle(
                overlay, (0, 0), (420, 40 + 30 * len(lane_stats)), (0, 0, 0), -1
            )
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

            y_offset_info = 30
            x_offset_info = 10
            if current_green_lane:
                cv2.putText(
                    frame,
                    f"GREEN: {current_green_lane}",
                    (x_offset_info, y_offset_info),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    LANE_COLORS["Low"],
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    "GREEN: None (Deciding...)",
                    (x_offset_info, y_offset_info),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    LANE_COLORS["Default"],
                    2,
                    cv2.LINE_AA,
                )
            y_offset_info += 40

            for roi_name, stats in lane_stats.items():
                congestion_level = stats["congestion"]
                count = stats["count"]
                color = LANE_COLORS.get(congestion_level, LANE_COLORS["Default"])
                text = f"{roi_name}: {count} vehicles ({congestion_level})"
                cv2.putText(
                    frame,
                    text,
                    (x_offset_info, y_offset_info),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                y_offset_info += 30

            # Draw ROI outlines
            for roi_name, poly_points in rois_display.items():
                color = LANE_COLORS.get(
                    lane_stats[roi_name]["congestion"], LANE_COLORS["Default"]
                )
                cv2.polylines(frame, [poly_points], True, color, 2)

            cv2.imshow("ROI Vehicle Counter", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        print("Resources released.")


if __name__ == "__main__":
    # url = "https://www.youtube.com/watch?v=ByED80IKdIU" //US URL
    url = "https://www.youtube.com/watch?v=CdesNZ150HE"
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    roi_labeler_dir = os.path.join(current_script_dir, "..", "roi_labeler")
    ROI_FILE_PATH = os.path.join(roi_labeler_dir, "road_rois.json")

    print(f"Using ROI file: {ROI_FILE_PATH}")

    if "youtube.com" in url or "youtu.be" in url:
        print("Getting direct stream URL from YouTube...")
        stream_url = get_stream_url(url)
        if stream_url:
            print("Stream URL found:", stream_url)
            main(stream_url, ROI_FILE_PATH)
        else:
            print("Could not retrieve stream URL. Exiting.")
    else:
        main(url, ROI_FILE_PATH)
