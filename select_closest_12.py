import json
import math

def select_closest_12():
    try:
        with open('d:/FYP/edgeagent/target_cameras.json', 'r') as f:
            cameras = json.load(f)
    except FileNotFoundError:
        print("target_cameras.json not found.")
        return

    if not cameras:
        print("No cameras found.")
        return

    print(f"Total cameras available: {len(cameras)}")

    # Calculate Centroid
    lats = [c['latitude'] for c in cameras if c.get('latitude')]
    lons = [c['longitude'] for c in cameras if c.get('longitude')]

    if not lats or not lons:
        print("No valid coordinates found.")
        return

    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)

    print(f"Centroid: ({centroid_lat}, {centroid_lon})")

    # Calculate distance squared (good enough for sorting) to centroid
    # treating lat/lon as cartesian roughly for small area
    for c in cameras:
        d_lat = c['latitude'] - centroid_lat
        d_lon = c['longitude'] - centroid_lon
        c['_dist_sq'] = d_lat**2 + d_lon**2

    # Sort and pick top 12
    cameras.sort(key=lambda x: x['_dist_sq'])
    selected = cameras[:12]

    # Remove temporary field
    for c in selected:
        del c['_dist_sq']

    print(f"Selected {len(selected)} closest cameras:")
    for c in selected:
        print(f" - {c['cameraName']}")

    # Save back
    with open('d:/FYP/edgeagent/target_cameras.json', 'w') as f:
        json.dump(selected, f, indent=2)

if __name__ == "__main__":
    select_closest_12()
