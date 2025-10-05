# swing_features.py
import numpy as np
from utils import line_angle, angle_3pt, smooth_series

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    out = landmarks.copy()
    T = landmarks.shape[0]
    for t in range(T):
        L = out[t]
        left_sh, right_sh = L[11][:2], L[12][:2]
        left_hip, right_hip = L[23][:2], L[24][:2]
        shoulder_w = np.linalg.norm(np.array(right_sh)-np.array(left_sh)) + 1e-6
        pelvis = (np.array(left_hip)+np.array(right_hip))/2
        out[t, :, 0] = (out[t, :, 0]-pelvis[0])/shoulder_w
        out[t, :, 1] = (out[t, :, 1]-pelvis[1])/shoulder_w
    return out

def detect_keyframes(landmarks: np.ndarray, fps: int = 30) -> dict:
    T = landmarks.shape[0]
    left_wr, right_wr = landmarks[:, 15, :2], landmarks[:, 16, :2]
    pos = (left_wr + right_wr)/2
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    vel_s = smooth_series(vel, window=7)
    if len(vel_s) == 0: return {"top": 0, "impact": T-1}
    impact_idx = int(np.argmax(vel_s)+1)
    half = max(1, T//2)
    top_idx = int(np.argmax(pos[:half,1]))
    return {"top": top_idx, "impact": impact_idx}

def extract_summary_features(landmarks: np.ndarray, fps: int = 30) -> dict:
    T = landmarks.shape[0]
    kf = detect_keyframes(landmarks, fps)
    top, impact = kf["top"], kf["impact"]
    
    def L(t, idx): return landmarks[t, idx, :2]

    shoulder_angle_top = line_angle(L(top, 11), L(top, 12))
    hip_angle_top = line_angle(L(top, 23), L(top, 24))
    x_factor = shoulder_angle_top - hip_angle_top

    left_elb = angle_3pt(L(impact,11), L(impact,13), L(impact,15))
    right_elb = angle_3pt(L(impact,12), L(impact,14), L(impact,16))
    elbow_at_impact = left_elb if abs(180-left_elb)<abs(180-right_elb) else right_elb

    t_back = max(1, top)/fps
    t_down = max(1, impact-top)/fps
    tempo_ratio = (t_back/t_down) if t_down>0 else float('inf')

    nose_y_series = landmarks[:,0,1]
    head_std = float(np.std(nose_y_series))

    left_wr, right_wr = landmarks[:,15,:2], landmarks[:,16,:2]
    hand_pos = (left_wr+right_wr)/2
    xs, ys = hand_pos[:,0], hand_pos[:,1]
    if np.ptp(xs)<1e-6:
        hand_dev = 0.0
    else:
        p = np.polyfit(xs, ys, 1)
        hand_dev = float(np.mean(np.abs(ys-np.polyval(p, xs))))

    features = {
        "shoulder_angle_top": shoulder_angle_top,
        "hip_angle_top": hip_angle_top,
        "x_factor_top": x_factor,
        "elbow_at_impact": elbow_at_impact,
        "tempo_ratio": tempo_ratio,
        "head_std": head_std,
        "hand_path_dev": hand_dev,
        "top_frame": top,
        "impact_frame": impact,
        "frames": T
    }
    return features
