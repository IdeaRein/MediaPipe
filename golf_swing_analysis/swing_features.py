import numpy as np
import cv2
import os
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
# 🏌️ 追加: ボール消失検出 (OpenCV)
# =========================================================
def detect_ball_disappearance(video_path: str) -> int | None:
    """
    OpenCVでボールが最後に見えるフレームを検出。
    明るい円形領域（白いボール）が見つかった最後のフレーム番号を返す。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ 動画が開けません: {video_path}")
        return None

    frame_idx = 0
    last_visible = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ボール（明るく小さな円）を検出
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=80, param2=18, minRadius=2, maxRadius=8
        )
        if circles is not None:
            last_visible = frame_idx
        frame_idx += 1

    cap.release()
    return last_visible


# =========================================================
# 🧩 改良版キーイベント検出（ボール補正対応）
# =========================================================
def detect_keyframes(landmarks: np.ndarray, fps: int = 30, video_path: str = None, output_dir: str = "outputs") -> dict:
    """
    改良版（トップ検出タイミング補正＋画像保存つき）:
      - トップ: 手の高さ(y)が上昇→下降に転じる“最高点”の直前
      - インパクト: 手速度が最大になる瞬間
      - 検出直後に対象フレーム画像を保存（video_path必須）
    """
    T = landmarks.shape[0]
    if T < 5:
        return {"top": 0, "impact": max(0, T - 1)}

    left_wr, right_wr = landmarks[:, 15, :3], landmarks[:, 16, :3]
    pos = (left_wr + right_wr) / 2.0  # 両手の中点
    y = pos[:, 1]
    z = pos[:, 2]

    # 🎯 1. 手の高さ変化（Y軸の速度）で上昇→下降の転換点を探す
    dy = np.gradient(y)
    dy_s = np.convolve(dy, np.ones(5)/5, mode='same')  # 平滑化
    top_candidates = np.where((dy_s[:-1] > 0) & (dy_s[1:] <= 0))[0]  # 上昇→下降の変化点

    if len(top_candidates) > 0:
        score = -y[top_candidates] + 0.3 * z[top_candidates]
        top_idx = int(top_candidates[np.argmax(score)])
    else:
        top_idx = int(np.argmin(y))

    # 🎯 2. インパクト検出：手の移動速度が最大になる瞬間
    vel = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * fps
    vel_s = np.convolve(vel, np.ones(5)/5, mode='same')
    impact_idx = int(np.argmax(vel_s))

    # 🎯 3. インパクト補正
    impact_idx = max(top_idx + 1, impact_idx)
    impact_idx = min(impact_idx, T - 2)

    # =========================================================
    # 🖼️ 4. 検出したフレームを画像として保存（video_pathがある場合のみ）
    # =========================================================
    if video_path is not None and os.path.exists(video_path):
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if top_idx < total_frames and impact_idx < total_frames:
            frame_indices = [top_idx, impact_idx]
            names = ["top_detect_frame.jpg", "impact_detect_frame.jpg"]
            for idx, name in zip(frame_indices, names):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    save_path = os.path.join(output_dir, name)
                    cv2.imwrite(save_path, frame)
                    print(f"🖼️ 保存完了: {name}（フレーム {idx}）")
        cap.release()
    else:
        print("⚠️ video_pathが指定されていないため、画像保存をスキップしました。")

    return {"top": top_idx, "impact": impact_idx}

# =========================================================
# 🧩 特徴量抽出
# =========================================================
def extract_summary_features(landmarks: np.ndarray, fps: int = 30, video_path: str = None, output_dir: str = "outputs") -> dict:
    T = landmarks.shape[0]
    kf = detect_keyframes(landmarks, fps, video_path, output_dir)
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
