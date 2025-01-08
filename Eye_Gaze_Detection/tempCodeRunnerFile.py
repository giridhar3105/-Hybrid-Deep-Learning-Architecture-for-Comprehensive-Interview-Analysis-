import cv2 as cv
import time
from datetime import datetime
from utils import analyze_behavior, save_data_to_csv, enhance_quality

# Load Haar cascade classifiers
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

# Initialize variables
blink_count = 0
gaze_movement = 0
prev_eye_position = None
start_time = time.time()
data = []

# Open a video capture stream
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be accessed. Exiting...")
    exit()

# Enhance capture quality
cap = enhance_quality(cap)

# Define VideoWriter to save the output
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
out = cv.VideoWriter('eye_gaze_analysis.avi', fourcc, 20.0, (640, 480))

gui_available = True  # Flag to check if GUI functions are working

try:
    cv.imshow("Test Window", cv.imread("test.jpg"))  # Test GUI functionality
    cv.destroyAllWindows()
except cv.error:
    gui_available = False
    print("GUI functionality is not supported. Skipping camera display...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame. Exiting...")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        eye_positions = []

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_center = (ex + ew // 2, ey + eh // 2)
            eye_positions.append(eye_center)

        # Analyze gaze movement
        if prev_eye_position and eye_positions:
            for eye in eye_positions:
                gaze_movement += abs(eye[0] - prev_eye_position[0]) + abs(eye[1] - prev_eye_position[1])

        prev_eye_position = eye_positions[0] if eye_positions else prev_eye_position

        # Blink detection (if eyes are not detected)
        if len(eyes) == 0:
            blink_count += 1

    # Save the processed frame to the video file
    out.write(frame)

    # Record analysis every 10 seconds
    if time.time() - start_time >= 10:
        behavior = analyze_behavior(blink_count, gaze_movement)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.append([timestamp, blink_count, gaze_movement, behavior])
        print(f"Recorded: {timestamp}, Blink Count: {blink_count}, Gaze Movement: {gaze_movement}, Behavior: {behavior}")
        blink_count = 0
        gaze_movement = 0
        start_time = time.time()

    # Display the frame if GUI is available
    if gui_available:
        cv.imshow("Eye Gaze Analysis", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break

# Release resources
cap.release()
out.release()
if gui_available:
    cv.destroyAllWindows()

# Save data to CSV
save_data_to_csv(data, "eye_gaze_analysis.csv")

print("Data saved for future analysis. Video saved as 'eye_gaze_analysis.avi'.")

