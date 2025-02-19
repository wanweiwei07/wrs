import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16

pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype)

cap = cv2.VideoCapture(0)
while(True):
    frame = cap.read()[1]
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        outputs = pipe.predict(frame)
        print(outputs)
        if outputs:
            outputs["keypoints"].draw_on_image(frame)
            cv2.imshow('frame', frame)
 
    k = cv2.waitKey(1)
    if k == 27:
        break
