import json

def filter_cameras():
    # Bounding Box
    MIN_LON = -122.343106
    MIN_LAT = 47.602499
    MAX_LON = -122.328635
    MAX_LAT = 47.612111

    print(f"Filtering cameras in bbox: Lat[{MIN_LAT}, {MAX_LAT}], Lon[{MIN_LON}, {MAX_LON}]")

    matched_cameras = []
    
    with open('d:/FYP/edgeagent/pa_cameras.txt', 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                cam_data = json.loads(line)
                lat = cam_data.get('latitude')
                lon = cam_data.get('longitude')
                
                if lat is None or lon is None:
                    continue
                    
                if MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON:
                    matched_cameras.append(cam_data)
            except Exception as e:
                print(f"Error processing line: {e}")
                continue
                    
    print(f"Found {len(matched_cameras)} matches.")
    
    # Save to a file for inspection
    with open('d:/FYP/edgeagent/target_cameras.json', 'w') as f:
        json.dump(matched_cameras, f, indent=2)
        
    for cam in matched_cameras:
        print(f" - {cam['cameraName']} ({cam['latitude']}, {cam['longitude']})")

if __name__ == "__main__":
    filter_cameras()
