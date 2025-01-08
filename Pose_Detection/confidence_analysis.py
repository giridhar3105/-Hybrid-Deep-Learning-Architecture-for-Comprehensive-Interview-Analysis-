import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def analyze_confidence(left_shoulder, right_shoulder, left_hip, right_hip):
    shoulder_distance = np.abs(left_shoulder[0] - right_shoulder[0])
    neck_angle = calculate_angle(
        left_shoulder,
        right_shoulder,
        [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2],
    )

    # Heuristics for confidence
    if shoulder_distance > 0.1 and neck_angle < 15:  # Adjust thresholds as needed
        return "Confident"
    else:
        return "Nervous"
