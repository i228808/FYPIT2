import cv2
import torch
import subprocess
import numpy as np
import yt_dlp
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import os
import pandas as pd


def get_stream_url(youtube_url):
    ydl_opts = {
        "format": "best[ext=mp4]",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=False)
        return info_dict["url"]


def main(stream_url, output_filename="output.mp4", duration=60, roi_labeler_dir=None):
    # YOLOv8 model loading
    try:
        model = YOLO("yolov9e.pt")  # Load a pre-trained YOLOv9e model
    except Exception as e:
        print("Error loading YOLOv8n model.")
        print(e)
        return

    # FFmpeg command to capture the stream
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        stream_url,
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    # Start the FFmpeg process
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

    # Get video properties for VideoWriter (assuming a constant frame rate, you might need to adjust this)
    # A more robust solution would involve getting frame rate from FFmpeg or the stream info
    fps = 30  # Assuming 30 fps, adjust if necessary
    frame_width = 1920
    frame_height = 1080

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can use other codecs like 'XVID'
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    vehicle_counts = []  # List to store vehicle counts per frame

    start_time = time.time()
    end_time = start_time + duration  # Run for specified duration

    while time.time() < end_time:
        # Read a frame from the FFmpeg process
        raw_frame = process.stdout.read(frame_width * frame_height * 3)
        if not raw_frame:
            break

        # Convert the raw frame to a NumPy array and create a writable copy
        frame = (
            np.frombuffer(raw_frame, np.uint8)
            .reshape((frame_height, frame_width, 3))
            .copy()
        )

        # Save the first frame as intersection.png in the roi_labeler directory
        if not hasattr(main, "first_frame_saved"):
            if roi_labeler_dir:
                intersection_image_path = os.path.join(
                    roi_labeler_dir, "intersection.png"
                )
                cv2.imwrite(intersection_image_path, frame)
                print(
                    f"Saved first frame as {intersection_image_path} for ROI labeling."
                )
            main.first_frame_saved = True

        # Perform inference
        results = model(frame)

        # Filter for vehicles and count them
        vehicle_count = 0
        # Check if results have boxes before iterating
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls = int(box.cls)
                # Check if the class is a vehicle (car, truck, motorcycle)
                if model.names[cls] in ["car", "truck", "motorcycle"]:
                    vehicle_count += 1
                    # Draw bounding box
                    xyxy = box.xyxy[0].tolist()
                    cv2.rectangle(
                        frame,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 255, 0),
                        2,
                    )

        # Append the vehicle count to the list
        vehicle_counts.append(vehicle_count)

        # Display the frame with vehicle count
        cv2.putText(
            frame,
            f"Vehicles: {vehicle_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # Write the annotated frame to the output video file
        out.write(frame)

        # Display the image in the notebook (optional, can be removed for faster processing)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        # In a Colab environment, cv2.waitKey will not work as expected.
        # I will comment out the break condition based on key press.
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        # Instead, you might want to break based on a different condition,
        # like processing a certain number of frames or a specific duration.
        # For this example, I'll add a simple break after one iteration for demonstration.
        # break # Remove this break to process multiple frames

    # Release the VideoWriter and close the FFmpeg process
    out.release()
    process.stdout.close()
    process.wait()
    # In a Colab environment, cv2.destroyAllWindows will not be necessary with cv2.imshow commented out.
    # cv2.destroyAllWindows()

    return vehicle_counts


if __name__ == "__main__":
    # Replace with your desired video stream URL
    url = "https://www.youtube.com/watch?v=ByED80IKdIU"  # e.g., a YouTube video URL
    output_filename = "annotated_video.mp4"
    duration = 60  # in seconds

    if "youtube.com" in url or "youtu.be" in url:
        print("Getting direct stream URL from YouTube...")
        url = get_stream_url(url)
        print("Stream URL found:", url)

        # Make sure the roi_labeler directory exists for saving the intersection.png
        roi_labeler_dir = "E:\\FYP\\edgeagent\\roi_labeler"
        if not os.path.exists(roi_labeler_dir):
            os.makedirs(roi_labeler_dir)

        vehicle_counts_data = main(url, output_filename, duration, roi_labeler_dir)

        print(f"Annotated video saved as {output_filename}")
