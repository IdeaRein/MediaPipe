import numpy as np
from utils import line_angle, angle_3pt, smooth_series


# =========================================================
# 🧩 ランドマーク正規化
# =========================================================
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    out = landmarks.copy()
    T = landmarks.shape[0]
    for t in range(T):
        L = out[t]
        left_sh, right_sh = L[11][:2], L[12][:2]
        left_hip, right_hip = L[23][:2], L[24][:2]
        shoulder_w = np.linalg.norm(np.array(right_sh) - np.array(left_sh)) + 1e-6
        pelvis = (np.array(left_hip) + np.array(right_hip)) / 2
        out[t, :, 0] = (out[t, :, 0] - pelvis[0]) / shoulder_w
        out[t, :, 1] = (out[t, :, 1] - pelvis[1]) / shoulder_w
    return out


# =========================================================
# 🧩 改良版キーイベント検出
# =========================================================
def _x_factor_series(landmarks: np.ndarray):
    """各フレームの肩角度・腰角度・Xファクター(肩-腰)を返す"""
    T = landmarks.shape[0]
    sh, hp, xf = [], [], []
    for t in range(T):
        L = landmarks[t]
        shoulder_angle = line_angle(L[11][:2], L[12][:2])
        hip_angle = line_angle(L[23][:2], L[24][:2])
        sh.append(shoulder_angle)
        hp.append(hip_angle)
        xf.append(shoulder_angle - hip_angle)
    return np.asarray(sh), np.asarray(hp), np.asarray(xf)


def detect_keyframes(landmarks: np.ndarray, fps: int = 30) -> dict:
    """
    改良版（後方撮影カメラ向け）:
      - トップ: 手が最も高く（y最小）かつ後方(z最大)の位置
      - インパクト: 手速度が急上昇し終える直前（加速度反転点）
    """
    T = landmarks.shape[0]
    if T < 5:
        return {"top": 0, "impact": max(0, T - 1)}

    left_wr, right_wr = landmarks[:, 15, :3], landmarks[:, 16, :3]
    pos = (left_wr + right_wr) / 2.0  # 両手の中点
    y, z = pos[:, 1], pos[:, 2]

    # ✅ トップ：前半区間で手が最も高く（yが最小）かつ後方(zが最大)
    half = max(1, T // 2)
    score_top = -y[:half] + z[:half] * 0.5  # y小さいほど上、z大きいほど後ろ
    top_idx = int(np.argmax(score_top))

    # ✅ インパクト：手の速度から加速度反転点を検出
    vel = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * fps
    vel_s = smooth_series(vel, window=5)
    acc = np.diff(vel_s)
    acc_s = smooth_series(acc, window=3)

    peak = int(np.argmax(vel_s))
    impact_idx = peak
    for i in range(max(1, peak - 8), peak + 1):
        if acc_s[i - 1] > 0 and acc_s[i] <= 0:
            impact_idx = i
            break

    impact_idx = int(np.clip(impact_idx, top_idx + 1, T - 2))
    return {"top": top_idx, "impact": impact_idx}


# =========================================================
# 🧩 特徴量抽出
# =========================================================
def extract_summary_features(landmarks: np.ndarray, fps: int = 30) -> dict:
    T = landmarks.shape[0]
    kf = detect_keyframes(landmarks, fps)
    top, impact = kf["top"], kf["impact"]

    # ✅ 追加：フレーム番号を出力
    print("========== フレーム検出情報 ==========")
    print(f"📍 最初のフレーム番号: 0")
    print(f"🏌️ トップフレーム番号: {top}")
    print(f"💥 インパクトフレーム番号: {impact}")
    print(f"🏁 最後のフレーム番号: {T - 1}")
    print("=====================================")

    def L(t, idx): return landmarks[t, idx, :2]

    shoulder_angle_top = line_angle(L(top, 11), L(top, 12))
    hip_angle_top = line_angle(L(top, 23), L(top, 24))
    x_factor = shoulder_angle_top - hip_angle_top

    left_elb = angle_3pt(L(impact, 11), L(impact, 13), L(impact, 15))
    right_elb = angle_3pt(L(impact, 12), L(impact, 14), L(impact, 16))
    elbow_at_impact = left_elb if abs(180 - left_elb) < abs(180 - right_elb) else right_elb

    t_back = max(1, top) / fps
    t_down = max(1, impact - top) / fps
    tempo_ratio = (t_back / t_down) if t_down > 0 else float('inf')

    nose_y_series = landmarks[:, 0, 1]
    head_std = float(np.std(nose_y_series))

    left_wr, right_wr = landmarks[:, 15, :2], landmarks[:, 16, :2]
    hand_pos = (left_wr + right_wr) / 2
    xs, ys = hand_pos[:, 0], hand_pos[:, 1]
    if np.ptp(xs) < 1e-6:
        hand_dev = 0.0
    else:
        p = np.polyfit(xs, ys, 1)
        hand_dev = float(np.mean(np.abs(ys - np.polyval(p, xs))))

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


# =========================================================
# 🏌️ 追加: 動的モーション解析ユーティリティ
# =========================================================
def refine_impact_by_motion(landmarks: np.ndarray, fps: int, rough_idx: int, window: int = 3) -> int:
    """手首速度＋前腕角速度を用いてインパクトを精密補正"""
    if landmarks.ndim != 3 or landmarks.shape[1] < 17:
        raise ValueError("landmarks 配列が不正です。shape=(frames,33,3) 形式を想定しています。")

    left_wr, right_wr = landmarks[:, 15, :2], landmarks[:, 16, :2]
    pos = (left_wr + right_wr) / 2
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    vel = np.pad(vel, (1, 0), mode='edge')
    vel = smooth_series(vel, window=5)

    left_elb = landmarks[:, 13, :2]
    left_forearm = left_wr - left_elb
    ang = np.unwrap(np.arctan2(left_forearm[:, 1], left_forearm[:, 0]))
    ang_vel = np.abs(np.gradient(ang) * fps)
    ang_vel = smooth_series(ang_vel, window=5)

    score = 0.6 * vel + 0.4 * ang_vel
    start = max(1, rough_idx - window)
    end = min(len(score) - 1, rough_idx + window)
    refined_idx = start + np.argmax(score[start:end])
    return int(refined_idx)


def compute_hand_speed(landmarks: np.ndarray, fps: int) -> np.ndarray:
    """各フレームの手首速度"""
    left_wr, right_wr = landmarks[:, 15, :2], landmarks[:, 16, :2]
    pos = (left_wr + right_wr) / 2
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    vel = np.pad(vel, (1, 0), mode='edge')
    return smooth_series(vel, window=5)


def compute_arm_angular_velocity(landmarks: np.ndarray, fps: int) -> np.ndarray:
    """左前腕の角速度"""
    left_elb = landmarks[:, 13, :2]
    left_wr = landmarks[:, 15, :2]
    left_forearm = left_wr - left_elb
    ang = np.unwrap(np.arctan2(left_forearm[:, 1], left_forearm[:, 0]))
    ang_vel = np.abs(np.gradient(ang) * fps)
    return smooth_series(ang_vel, window=5)
