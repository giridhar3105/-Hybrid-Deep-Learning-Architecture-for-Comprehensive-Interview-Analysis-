import cv2
import csv

def analyze_behavior(blink_count, gaze_movement):
    """Determine nervousness or confidence based on blink count and gaze movement."""
    if blink_count > 10 or gaze_movement > 20:
        return "Nervous"
    else:
        return "Confident"

def save_data_to_csv(data, filename):
    """Save data to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Blink Count", "Gaze Movement", "Behavior"])
        writer.writerows(data)

def enhance_quality(cap):
    """Enhance the quality of live video capture."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap