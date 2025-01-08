import cv2 as cv
import mediapipe as mp

class MediapipePose:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame):
        # Convert frame to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        landmarks = None
        if results.pose_landmarks:
            landmarks = {
                'left_shoulder': [
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ],
                'right_shoulder': [
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                ],
                'left_hip': [
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP.value].y,
                ],
                'right_hip': [
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                ],
            }
            # Draw landmarks
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return landmarks, frame
