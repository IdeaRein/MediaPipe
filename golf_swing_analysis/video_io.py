import cv2
import mediapipe as mp
import numpy as np
import os
from swing_features import extract_summary_features


# =========================================================
# 🟡 ボール静止検出（追加関数）
# =========================================================
def detect_ball_stationary_frame(ball_positions: np.ndarray, threshold=1.0, stable_count=5):
    """
    ボールの座標変化が一定以下の状態が連続したら「静止」と判定し、
    その最後のフレーム番号を返す。
    """
    if len(ball_positions) < 2:
        return None

    # NaN除去（欠損が混じっていると差分計算が壊れるため）
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
# 🎯 ボール追跡（OpenCV版）軽量トラッカー
# =========================================================
def track_ball_positions(video_path: str, resize_scale=0.5, color_thresh=(180, 30, 30), debug=False):
    """
    白いボール領域をフレームごとに追跡し、(x, y)座標を返す。
    照明条件によってはcolor_threshを調整する。
    """
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

        # ✅ 修正1: color_threshの使い方（H, S, Vごとのしきい値）
        lower_white = np.array([0, 0, 255 - color_thresh[2]])  # 255 - V成分
        upper_white = np.array([180, color_thresh[1], 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        # ✅ 修正2: ノイズ除去（誤検出防止）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(c)
            positions.append((x, y))
            if debug:
                cv2.circle(small, (int(x), int(y)), int(r), (0, 255, 0), 2)
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
# 🧩 ランドマーク抽出＋フレーム保存（既存拡張）
# =========================================================
def extract_landmarks_from_video(video_path, resize_scale=0.5, max_frames=1000, output_dir="outputs"):
    """
    動画からMediaPipe Poseで骨格ランドマークを抽出し、
    トップ・インパクト・最初・最後のフレームを保存。
    さらにボール座標も追跡して静止フレームを検出。
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"動画が開けません: {video_path}")

    # ✅ 修正3: FPSが取得できない動画対策
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30

    original_frames = []
    landmarks_all = []
    frame_indices = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        original_frames.append(frame.copy())
        h, w, _ = frame.shape
        small_frame = cv2.resize(frame, (int(w * resize_scale), int(h * resize_scale)))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        result = pose.process(frame_rgb)
        if result.pose_landmarks:
            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
            )
            landmarks_all.append(frame_landmarks)
            frame_indices.append(frame_count)

        frame_count += 1

    cap.release()
    pose.close()

    if len(landmarks_all) == 0:
        raise RuntimeError("ランドマークが検出されませんでした。")

    landmarks_all = np.array(landmarks_all)
    np.save(os.path.join(output_dir, "landmarks.npy"), landmarks_all)
    print(f"✅ 抽出フレーム数: {len(original_frames)}（検出成功: {len(landmarks_all)}）")

    # ---- トップ・インパクト検出 ----
    features = extract_summary_features(landmarks_all, fps=int(fps))
    top_idx, impact_idx = features["top_frame"], features["impact_frame"]
    top_real_idx = frame_indices[top_idx] if top_idx < len(frame_indices) else None
    impact_real_idx = frame_indices[impact_idx] if impact_idx < len(frame_indices) else None

    # ---- ボール追跡と静止判定 ----
    print("🎯 ボール追跡を開始します...")
    ball_positions = track_ball_positions(video_path, resize_scale=resize_scale)
    np.save(os.path.join(output_dir, "ball_positions.npy"), ball_positions)

    stationary_frame = detect_ball_stationary_frame(ball_positions)
    if stationary_frame is not None:
        print(f"⚪ ボール静止フレーム検出: {stationary_frame}")
    else:
        print("⚪ ボール静止フレームは検出されませんでした。")

    # ---- フレーム画像の保存 ----
    # ✅ 修正4: Noneチェック + 範囲外アクセス防止
    if len(original_frames) > 0:
        cv2.imwrite(os.path.join(output_dir, "first_frame.jpg"), original_frames[0])
        cv2.imwrite(os.path.join(output_dir, "last_frame.jpg"), original_frames[-1])
    if top_real_idx is not None and 0 <= top_real_idx < len(original_frames):
        cv2.imwrite(os.path.join(output_dir, "top_frame.jpg"), original_frames[top_real_idx])
    if impact_real_idx is not None and 0 <= impact_real_idx < len(original_frames):
        cv2.imwrite(os.path.join(output_dir, "impact_frame.jpg"), original_frames[impact_real_idx])

    print(f"✅ トップ・インパクト＋最初・最後フレームを保存しました ({output_dir})")

    return landmarks_all, ball_positions, stationary_frame
