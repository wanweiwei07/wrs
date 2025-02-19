import torch
import cv2
import numpy as np
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

# **Initialize Model**
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16
pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

# **Hand Keypoints Connection Pairs for Visualization**
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def compute_angle(v1, v2):
    """
    Compute the angle between two vectors using the cosine formula.
    :param v1: First vector (numpy array)
    :param v2: Second vector (numpy array)
    :return: Angle in degrees
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:  # Avoid division by zero
        return 0.0
    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.arccos(cos_theta)

def compute_selected_angles(keypoints):
    """
    Compute angles for selected joints in the hand skeleton.
    :param keypoints: 21 hand keypoints from WiLor Model
    :return: Dictionary of computed angles
    """
    joint_pairs = {
        "thumb0": (0, 1, 2),
        "thumb1": (1, 2, 3),
        "thumb2": (2, 3, 4),
        "index0": (10, 5, 6),
        "index1": (0, 5, 6),
        "index2": (5, 6, 7),
        "middle0": (0, 9, 10),
        "middle1": (9, 10, 11),
        "ring0": (0, 13, 14),
        "ring1": (13, 14, 15),
        "pinky0": (0,17,18),
        "pinky1": (17,18,19)
    }
    angles = {}
    for joint_name, (p1, p2, p3) in joint_pairs.items():
        v1 = np.array(keypoints[p1]) - np.array(keypoints[p2])  # Vector 1
        v2 = np.array(keypoints[p3]) - np.array(keypoints[p2])  # Vector 2
        angles[joint_name] = compute_angle(v1, v2)
        if joint_name=="index0":
            angles[joint_name] = np.pi/6-angles[joint_name]
        else:
            angles[joint_name] = np.pi-angles[joint_name]
    return angles


# **Capture Video from Webcam**
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img_h, img_w, _ = frame.shape
    if not ret:
        break
    # **Convert frame to RGB for model prediction**
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # **Run hand pose estimation**
    outputs = pipe.predict(rgb_frame)
    if len(outputs) > 0:
        for hand in outputs:
            if hand["is_right"]:
                keypoints = hand["wilor_preds"]["pred_keypoints_3d"]  # Shape: (N, 21, 3)  -> 3D Keypoints
                print(keypoints)
                angles = compute_selected_angles(keypoints[0])
                for joint, angle in angles.items():
                    print(f"{joint}: {np.degrees(angle):.2f}°")
                # for hand in keypoints:  # Iterate over detected hands
                #     for i, (x, y, z) in enumerate(hand):  # Loop through all 21 keypoints
                #         x = int(x * img_w) if x < 1 else int(x)
                #         y = int(y * img_h) if y < 1 else int(y)
                #         cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Draw keypoints
                #     # **Draw Hand Connections**
                #     for connection in HAND_CONNECTIONS:
                #         x1, y1, _ = hand[connection[0]]
                #         x2, y2, _ = hand[connection[1]]
                #         x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                #         cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                break
    # **Display the Frame**
    # cv2.imshow("Hand Pose Estimation", frame)
    # **Break Loop on 'q' Key**
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# **Cleanup**
cap.release()
cv2.destroyAllWindows()
