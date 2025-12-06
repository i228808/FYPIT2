import cv2
import json
import numpy as np
import os
import glob

# Constants
MAX_DISPLAY_WIDTH = 1280
MAX_DISPLAY_HEIGHT = 720
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "road_rois.json")

# Global state
current_image_path = ""
original_image = None
display_image = None
clone_display_image = None
image_for_drawing = None
scaling_factor = 1.0
current_points = []
current_rois = {} # ROIs for the current image
all_rois = {} # All ROIs loaded from file + current session

def load_rois():
    global all_rois
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            try:
                all_rois = json.load(f)
                print(f"Loaded {len(all_rois)} existing camera configs.")
            except json.JSONDecodeError:
                print("Error reading existing ROI file. Starting fresh.")
                all_rois = {}

def save_rois():
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_rois, f, indent=4)
    print(f"Saved ROIs to '{OUTPUT_FILE}'")

def scale_point_to_original(x_display, y_display):
    return (int(x_display / scaling_factor), int(y_display / scaling_factor))

def scale_point_to_display(x_original, y_original):
    return (int(x_original * scaling_factor), int(y_original * scaling_factor))

def redraw_current_drawing_state():
    global image_for_drawing
    image_for_drawing = clone_display_image.copy()

    # Redraw all saved ROIs for this image (scaled)
    for name, points in current_rois.items():
        if len(points) > 2:
            display_points = [scale_point_to_display(p[0], p[1]) for p in points]
            pts = np.array(display_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image_for_drawing, [pts], True, (255, 0, 0), 2)
            if display_points:
                cv2.putText(image_for_drawing, name, (display_points[0][0], display_points[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Redraw current points being drawn (green)
    if len(current_points) > 0:
        for i, p_orig in enumerate(current_points):
            p_display_x, p_display_y = scale_point_to_display(p_orig[0], p_orig[1])
            cv2.circle(image_for_drawing, (p_display_x, p_display_y), 5, (0, 255, 0), -1)
            if i > 0:
                prev_p_orig = current_points[i - 1]
                prev_p_display_x, prev_p_display_y = scale_point_to_display(prev_p_orig[0], prev_p_orig[1])
                cv2.line(image_for_drawing, (prev_p_display_x, prev_p_display_y), 
                         (p_display_x, p_display_y), (0, 255, 0), 2)

    cv2.imshow("ROI Labeler", image_for_drawing)

def click_event(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x, original_y = scale_point_to_original(x, y)
        current_points.append((original_x, original_y))
        redraw_current_drawing_state()

def process_image(image_file):
    global original_image, display_image, clone_display_image, image_for_drawing, scaling_factor, current_points, current_rois, current_image_path
    
    current_image_path = image_file
    camera_name = os.path.splitext(os.path.basename(image_file))[0]
    print(f"\nProcessing: {camera_name}")
    
    original_image = cv2.imread(image_file)
    if original_image is None:
        print(f"Could not load {image_file}")
        return

    # Load existing ROIs for this camera if any
    current_rois = all_rois.get(camera_name, {})
    current_points = []

    # Scaling
    (h, w) = original_image.shape[:2]
    scaling_factor = 1.0
    if w > MAX_DISPLAY_WIDTH or h > MAX_DISPLAY_HEIGHT:
        scale_w = MAX_DISPLAY_WIDTH / w
        scale_h = MAX_DISPLAY_HEIGHT / h
        scaling_factor = min(scale_w, scale_h)

    display_image = cv2.resize(original_image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    clone_display_image = display_image.copy()
    image_for_drawing = display_image.copy()

    cv2.namedWindow("ROI Labeler")
    cv2.setMouseCallback("ROI Labeler", click_event)
    
    redraw_current_drawing_state()

    print("  Controls:")
    print("  [Enter]: Start new ROI (will ask for name)")
    print("  [n]: Finish current ROI")
    print("  [z]: Undo last point")
    print("  [s]: Save and Next Image")
    print("  [q]: Quit")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == 13: # Enter
            roi_name = input("  Enter ROI name (e.g., 'Lane 1', 'Northbound'): ").strip()
            if roi_name:
                print(f"  Drawing ROI: {roi_name}")
                current_points = []
                # Temporarily store name to associate when 'n' is pressed
                # But simple approach: just start drawing, 'n' saves to this name
                # We need to store the name somewhere. Let's use a global or just ask after drawing?
                # Better: Ask BEFORE drawing as per original script logic, but let's stick to the flow.
                # Actually, original script asked name then cleared points.
                # Let's follow: Enter -> Name -> Draw -> 'n' to save.
                
                # To support multiple ROIs per image, we need to know which one we are drawing.
                # Let's simplify: 
                # 1. User clicks points.
                # 2. User presses 'n'.
                # 3. System asks for name.
                pass # Logic handled in 'n'

        elif key == ord('n'):
            if len(current_points) > 2:
                roi_name = input("  Enter name for this ROI: ").strip()
                if roi_name:
                    current_rois[roi_name] = current_points.copy()
                    print(f"  Added ROI: {roi_name}")
                    current_points = []
                    redraw_current_drawing_state()
                else:
                    print("  Name required. ROI not saved.")
            else:
                print("  Need at least 3 points.")

        elif key == ord('z'):
            if current_points:
                current_points.pop()
                redraw_current_drawing_state()

        elif key == ord('s'): # Save and Next
            all_rois[camera_name] = current_rois
            save_rois()
            return True # Continue to next image

        elif key == ord('q'): # Quit
            all_rois[camera_name] = current_rois
            save_rois()
            return False # Stop processing

def main():
    load_rois()
    
    if not os.path.exists(IMAGES_DIR):
        print(f"Images directory not found: {IMAGES_DIR}")
        return

    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
    image_files.sort()
    
    if not image_files:
        print("No images found to label.")
        return

    print(f"Found {len(image_files)} images.")
    
    for img_file in image_files:
        if not process_image(img_file):
            break
            
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
