import torch
import cv2
import numpy as np
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from wrs import wd, rm, mcm
from wrs.robot_sim.end_effectors.multifinger.xhand import xhand_right as xhr

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

def map_angle(x, src_min, src_max, dst_min, dst_max, reverse=True):
    if x > src_max:
        x = src_max
    if x < src_min:
        x = src_min
    if reverse:
        return (x - src_min) / (src_max - src_min) * (dst_min - dst_max) + dst_max
    else:
        return (x - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min

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
        if joint_name=="thumb0":
            angles[joint_name] = map_angle(angles[joint_name], 2.4, 2.9, 0, 1.57)
        if joint_name=="thumb1":
            angles[joint_name] = map_angle(angles[joint_name], 2.5, 3.0, 0, 1.0)
        if joint_name=="thumb2":
            angles[joint_name] = map_angle(angles[joint_name], 2.3, 2.9, 0, 1.57)
        if joint_name=="index0":
            angles[joint_name] = map_angle(angles[joint_name], 0.5, 1.0,  -0.087, 0.297, reverse = False)
        if joint_name=="index1":
            angles[joint_name] = map_angle(angles[joint_name], 2.0, 2.9, 0, 1.5)
        if joint_name=="index2":
            angles[joint_name] = map_angle(angles[joint_name], 1.6, 3, 0, 1.92)
        if joint_name=="middle0":
            angles[joint_name] = map_angle(angles[joint_name], 2.0, 2.9, 0, 1.5)
        if joint_name=="middle1":
            angles[joint_name] = map_angle(angles[joint_name], 1.6, 2.9, 0, 1.92)
        if joint_name=="ring0":
            angles[joint_name] = map_angle(angles[joint_name], 2.0, 2.9, 0, 1.5)
        if joint_name=="ring1":
            angles[joint_name] = map_angle(angles[joint_name], 1.6, 2.8, 0, 1.92)
        if joint_name=="pinky0":
            angles[joint_name] = map_angle(angles[joint_name], 2.0, 2.9, 0, 1.5)
        if joint_name=="pinky1":
            angles[joint_name] = map_angle(angles[joint_name], 1.6, 2.9, 0, 1.92)
    return angles

base = wd.World(cam_pos=rm.vec(1.7, 1.7, 1.7), lookat_pos=rm.vec(0, 0, .3))
xhand = xhr.XHandRight(pos=rm.vec(0, 0, 0), rotmat=rm.rotmat_from_euler(0, 0, 0))

# **Capture Video from Webcam**
cap = cv2.VideoCapture(0)
onscreen_list= []
def update(rbt, cap, onscreen_list, task):
    ret, frame = cap.read()
    img_h, img_w, _ = frame.shape
    if not ret:
        return task.cont
    # **Convert frame to RGB for model prediction**
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # **Run hand pose estimation**
    outputs = pipe.predict(rgb_frame)
    if len(outputs) > 0:
        for hand in outputs:
            if hand["is_right"]:
                keypoints = hand["wilor_preds"]["pred_keypoints_3d"]  # Shape: (N, 21, 3)  -> 3D Keypoints
                angles = compute_selected_angles(keypoints[0])
                for joint, angle in angles.items():
                    print(f"{joint}: {angle:.2f}°")
                xhand.goto_given_conf(rm.np.array(list(angles.values())))
                for ele in onscreen_list:
                    ele.detach()
                onscreen_list.append(xhand.gen_meshmodel())
                onscreen_list[-1].attach_to(base)
            break

    return task.cont

taskMgr.doMethodLater(0.01, update, "update", extraArgs=[xhand, cap, onscreen_list], appendTask=True)
base.run()
# **Cleanup**
cap.release()
cv2.destroyAllWindows()
