import os
from pymongo import MongoClient
from datetime import datetime

class DatabaseManager:
    def __init__(self, uri=None, db_name="traffic_monitoring"):
        if uri is None:
            uri = os.getenv("MONGODB_URI", "mongodb+srv://Abdullah:1968@cluster0.zaumwkl.mongodb.net/?appName=Cluster0")
        
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[db_name]
            self.metrics_collection = self.db["traffic_metrics"]
            # Test connection
            self.client.server_info()
            print(f"Connected to MongoDB: {db_name}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None

    def insert_frame_metrics(self, camera_name, timestamp, vehicle_counts, congestion_levels):
        """
        Inserts metrics for a single frame.
        
        Args:
            camera_name (str): Name of the camera.
            timestamp (float): Unix timestamp.
            vehicle_counts (dict): Dictionary of ROI names to vehicle counts.
            congestion_levels (dict): Dictionary of ROI names to congestion levels.
        """
        if self.db is None:
            return

        document = {
            "camera_name": camera_name,
            "timestamp": datetime.fromtimestamp(timestamp),
            "vehicle_counts": vehicle_counts,
            "congestion_levels": congestion_levels,
            "total_vehicles": sum(vehicle_counts.values())
        }
        
        try:
            self.metrics_collection.insert_one(document)
        except Exception as e:
            print(f"Error inserting metrics: {e}")

    def close(self):
        if self.client:
            self.client.close()
