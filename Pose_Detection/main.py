import cv2 as cv
import time
from mediapipe_pose import MediapipePose
from confidence_analysis import analyze_confidence
from csv_writer import save_to_csv
from model_handler import save_model, load_model

# Initialize Mediapipe Pose
pose_model = MediapipePose()

# Variables for analysis
data_points = []
analysis_start_time = time.time()
analysis_duration = 300  # 5 minutes for demo
trained_model_path = "pose_analysis_model.pkl"

# Load an existing model if available
trained_model = load_model(trained_model_path)
if not trained_model:
    trained_model = {}  # Initialize a new model if none exists

# Open webcam feed
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error accessing webcam.")
        break

    # Process the frame with Mediapipe
    landmarks, annotated_frame = pose_model.process_frame(frame)

    if landmarks:
        # Get required coordinates
        left_shoulder = landmarks['left_shoulder']
        right_shoulder = landmarks['right_shoulder']
        left_hip = landmarks['left_hip']
        right_hip = landmarks['right_hip']

        # Analyze confidence
        confidence_status = analyze_confidence(left_shoulder, right_shoulder, left_hip, right_hip)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        data_points.append({"Timestamp": timestamp, "Status": confidence_status})

        # Store in model for future reference
        trained_model[timestamp] = confidence_status

        # Visualize the confidence status
        cv.putText(
            annotated_frame,
            f"Status: {confidence_status}",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0) if confidence_status == "Confident" else (0, 0, 255),
            2,
        )

    # Display the frame
    cv.imshow('Pose Confidence Analysis', annotated_frame)

    # Exit on 'q' key or when the analysis duration ends
    if cv.waitKey(10) & 0xFF == ord('q') or time.time() - analysis_start_time > analysis_duration:
        break

# Save results to a CSV file
save_to_csv(data_points)

# Save the trained model
save_model(trained_model, trained_model_path)

# Release resources
cap.release()
cv.destroyAllWindows()
