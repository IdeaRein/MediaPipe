# video_io.py
import cv2
import mediapipe as mp
import numpy as np

def extract_landmarks_from_video(video_path, resize_scale=0.5, max_frames=1000):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けません: {video_path}")
    
    landmarks_all = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        
        h, w, _ = frame.shape
        small_frame = cv2.resize(frame, (int(w*resize_scale), int(h*resize_scale)))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
            )
            landmarks_all.append(frame_landmarks)
        
        frame_count += 1
    
    cap.release()
    return np.array(landmarks_all)  # (T, 33, 4)
