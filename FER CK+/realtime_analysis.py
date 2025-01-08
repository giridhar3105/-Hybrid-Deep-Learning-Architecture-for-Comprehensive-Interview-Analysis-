import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# Load the pre-trained model
model = load_model('cnn_model.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels (adjust based on your model's output order)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize data logging
emotion_counts = {emotion: 0 for emotion in emotion_labels}
start_time = datetime.now()
duration = timedelta(minutes=1)  # 1-minute analysis

# Open webcam feed
cap = cv2.VideoCapture(0)  # 0 for the default webcam

# Prepare CSV file for logging
csv_file = 'emotion_analysis.csv'
df = pd.DataFrame(columns=['Timestamp', 'Emotion'])
df.to_csv(csv_file, index=False)

print("Starting real-time emotion recognition...")
while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract face ROI
        face = gray[y:y+h, x:x+w]
        # Resize to match model input size (adjust size based on your model)
        face_resized = cv2.resize(face, (48, 48))  # FER models often use 48x48 grayscale images
        face_normalized = face_resized / 255.0  # Normalize pixel values
        face_reshaped = np.reshape(face_normalized, (1, 48, 48, 1))  # Add batch and channel dimensions

        # Predict emotion
        prediction = model.predict(face_reshaped, verbose=0)
        emotion_index = np.argmax(prediction)
        emotion_label = emotion_labels[emotion_index]

        # Increment emotion count
        emotion_counts[emotion_label] += 1

        # Log data into CSV
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.DataFrame([[timestamp, emotion_label]], columns=['Timestamp', 'Emotion'])
        df.to_csv(csv_file, mode='a', header=False, index=False)

        # Draw rectangle around the face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Facial Expression Recognition', frame)

    # Check if one minute has passed
    if datetime.now() - start_time >= duration:
        break

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Analyze emotion counts
most_frequent_emotion = max(emotion_counts, key=emotion_counts.get)
print("\nEmotion Analysis Summary (1 Minute):")
for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count} times")
print(f"Most Frequent Emotion: {most_frequent_emotion}")

# Save final counts to CSV
df_summary = pd.DataFrame.from_dict(emotion_counts, orient='index', columns=['Count'])
df_summary.to_csv('emotion_summary.csv')

print("\nAnalysis complete. Results saved to 'emotion_summary.csv'.")
