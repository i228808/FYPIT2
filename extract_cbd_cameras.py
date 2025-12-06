import json

def extract_cbd_cameras():
    target_streets = [
        "University St", 
        "Seneca St", 
        "Madison St", 
        "Columbia St", 
        "Cherry St",
        "Spring St", # Added generic CBD streets based on previous context if needed, but sticking to user list first
        "James St",
        "Marion St",
        "Pike St", # Central
        "Pine St", # Central
        "Union St"
    ]
    # User strictly asked for: University, Seneca, Madison, Columbia, Cherry. 
    # But "etc" implies more. I will prioritize the user's list.
    
    keywords = ["University", "Seneca", "Madison", "Columbia", "Cherry"]
    
    print(f"Searching for cameras with keywords: {keywords}")

    matched_cameras = []
    seen_names = set()
    
    with open('d:/FYP/edgeagent/pa_cameras.txt', 'r') as f:
        for line in f:
            if not line.strip(): continue
            cam_data = json.loads(line)
            name = cam_data.get('cameraName')
            if not name:
                continue

            # We want to be careful not to pick up distant cameras if street names differ, 
            # but usually "StreetName & StreetName" is the format.
            
            for kw in keywords:
                if kw in name:
                    if name not in seen_names:
                        matched_cameras.append(cam_data)
                        seen_names.add(name)
                    break 
                    
    print(f"Found {len(matched_cameras)} matches.")
    
    # Save to a file for inspection
    with open('d:/FYP/edgeagent/target_cameras.json', 'w') as f:
        json.dump(matched_cameras, f, indent=2)
        
    for cam in matched_cameras:
        print(f" - {cam['cameraName']}")

if __name__ == "__main__":
    extract_cbd_cameras()
