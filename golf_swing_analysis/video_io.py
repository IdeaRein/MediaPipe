import cv2
import mediapipe as mp
import numpy as np
import os
from swing_features import extract_summary_features


# =========================================================
# 🟡 ボール静止検出
# =========================================================
def detect_ball_stationary_frame(ball_positions: np.ndarray, threshold=1.0, stable_count=5):
    """ボールの座標変化が一定以下の状態が連続したら「静止」と判定。"""
    if len(ball_positions) < 2:
        return None

    valid = ~np.isnan(ball_positions[:, 0])
    ball_positions = ball_positions[valid]
    if len(ball_positions) < 2:
        return None

    distances = np.linalg.norm(np.diff(ball_positions, axis=0), axis=1)
    stable_frames = np.where(distances < threshold)[0]

    for i in range(len(stable_frames) - stable_count):
        if stable_frames[i + stable_count - 1] - stable_frames[i] == stable_count - 1:
            return int(stable_frames[i + stable_count - 1])
    return None


# =========================================================
# 🎯 ボール追跡（OpenCV版）
# =========================================================
def track_ball_positions(video_path: str, resize_scale=0.5, color_thresh=(180, 30, 30), debug=False):
    """白いボール領域をフレームごとに追跡し、(x, y)座標を返す。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けません: {video_path}")

    positions = []
    kernel = np.ones((3, 3), np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        small = cv2.resize(frame, (int(w * resize_scale), int(h * resize_scale)))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 255 - color_thresh[2]])
        upper_white = np.array([180, color_thresh[1], 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(c)
            positions.append((x, y))
        else:
            positions.append((np.nan, np.nan))

        if debug:
            cv2.imshow("ball_tracking", small)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    return np.array(positions)


# =========================================================
# 🧩 ランドマーク抽出＋フレーム保存（ズレ完全修正版）
# =========================================================
def extract_landmarks_from_video(video_path, resize_scale=0.5, max_frames=1000, output_dir="outputs"):
    """
    動画からMediaPipe Poseで骨格ランドマークを抽出し、
    トップ・インパクト・最初・最後・全フレームを正確に保存。
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けません: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30

    frame_count = 0
    landmarks_all = []
    all_frames = []       # 全フレーム
    detected_frames = []  # ポーズ検出に成功したフレーム（これを保存に使う）

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        all_frames.append(frame.copy())  # 全フレーム保存
        h, w, _ = frame.shape
        small_frame = cv2.resize(frame, (int(w * resize_scale), int(h * resize_scale)))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
            )
            landmarks_all.append(frame_landmarks)
            detected_frames.append(frame.copy())  # ← 検出成功フレームを保存に使う

        frame_count += 1

    cap.release()
    pose.close()

    if not landmarks_all:
        raise RuntimeError("ランドマークが検出されませんでした。")

    landmarks_all = np.array(landmarks_all)
    np.save(os.path.join(output_dir, "landmarks.npy"), landmarks_all)
    print(f"✅ 抽出フレーム数: {len(all_frames)}（検出成功: {len(landmarks_all)}）")

    # ===== トップ・インパクト検出 =====
    features = extract_summary_features(landmarks_all, fps=int(fps))
    top_idx, impact_idx = features["top_frame"], features["impact_frame"]
    print(f"🔍 検出結果 → トップ:{top_idx}, インパクト:{impact_idx}")

    # ===== 全フレーム出力 =====
    for i, f in enumerate(all_frames):
        cv2.imwrite(os.path.join(output_dir, f"frame_{i:03d}.jpg"), f)
    print(f"🖼️ 全 {len(all_frames)} フレーム保存完了 ({output_dir})")

    # ===== 代表フレーム保存（完全一致版） =====
    # トップとインパクトは検出に成功したフレームから直接保存（ズレなし）
    if 0 <= top_idx < len(detected_frames):
        cv2.imwrite(os.path.join(output_dir, "top_frame.jpg"), detected_frames[top_idx])
    else:
        print(f"⚠️ トップフレーム番号 {top_idx} が範囲外です")

    if 0 <= impact_idx < len(detected_frames):
        cv2.imwrite(os.path.join(output_dir, "impact_frame.jpg"), detected_frames[impact_idx])
    else:
        print(f"⚠️ インパクトフレーム番号 {impact_idx} が範囲外です")

    # 最初・最後は全フレームから
    if all_frames:
        cv2.imwrite(os.path.join(output_dir, "first_frame.jpg"), all_frames[0])
        cv2.imwrite(os.path.join(output_dir, "last_frame.jpg"), all_frames[-1])

    print("✅ トップ・インパクト・最初・最後フレームを保存完了！")

    # ===== ボール追跡 =====
    print("🎯 ボール追跡を開始します...")
    ball_positions = track_ball_positions(video_path, resize_scale=resize_scale)
    stationary_frame = detect_ball_stationary_frame(ball_positions)
    np.save(os.path.join(output_dir, "ball_positions.npy"), ball_positions)

    if stationary_frame is not None:
        print(f"⚪ ボール静止フレーム検出: {stationary_frame}")
    else:
        print("⚪ ボール静止フレームは検出されませんでした。")

    return landmarks_all, ball_positions, stationary_frame
