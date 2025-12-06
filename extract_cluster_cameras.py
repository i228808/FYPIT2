import json

def extract_cameras():
    # Load clusters
    with open('d:/FYP/edgeagent/camera_clusters.json', 'r') as f:
        clusters = json.load(f)
    
    target_cluster = None
    # Find cluster with exactly 10 cameras
    for c in clusters:
        if c['camera_count'] == 10:
            target_cluster = c
            break
            
    if not target_cluster:
        print("Cluster with exactly 10 cameras not found. Searching for closest...")
        # Find closest to 10
        clusters.sort(key=lambda x: abs(x['camera_count'] - 10))
        target_cluster = clusters[0]
        print(f"Selected cluster {target_cluster['cluster_id']} with {target_cluster['camera_count']} cameras.")

    print(f"Found cluster {target_cluster['cluster_id']} with {len(target_cluster['cameras'])} cameras")
    
    # Get camera names from cluster
    cluster_camera_names = set()
    for cam in target_cluster['cameras']:
        if cam['cameraName']:
            cluster_camera_names.add(cam['cameraName'])
            
    print(f"Cluster has {len(cluster_camera_names)} named cameras")

    # Find matching cameras in pa_cameras.txt
    matched_cameras = []
    with open('d:/FYP/edgeagent/pa_cameras.txt', 'r') as f:
        for line in f:
            if not line.strip(): continue
            cam_data = json.loads(line)
            if cam_data['cameraName'] in cluster_camera_names:
                matched_cameras.append(cam_data)
                
    print(f"Found {len(matched_cameras)} matching cameras in pa_cameras.txt")
    
    # Save to a file for inspection
    with open('d:/FYP/edgeagent/target_cameras.json', 'w') as f:
        json.dump(matched_cameras, f, indent=2)

if __name__ == "__main__":
    extract_cameras()
