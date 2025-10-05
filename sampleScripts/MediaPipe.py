import cv2
import mediapipe as mp
import numpy as np
import json
from swing_scoring import process_landmarks_and_score, save_result_json, save_result_npz

# ============ ランドマーク抽出部分 ============

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
        
        height, width, _ = frame.shape
        small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))
        frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        result = pose.process(frame_rgb)
        
        if result.pose_landmarks:
            # (33, 4) の配列を作成 (x,y,z,visibility)
            frame_landmarks = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
            )
            landmarks_all.append(frame_landmarks)
        
        frame_count += 1
    
    cap.release()
    return np.array(landmarks_all)  # shape = (T, 33, 4)


# ============ メイン処理部分 ============

def analyze_golf_swing(video_path, output_prefix="swing_result"):
    print(f"動画処理中: {video_path}")
    
    # 1. 動画からランドマーク抽出
    landmarks = extract_landmarks_from_video(video_path)
    print(f"抽出フレーム数: {landmarks.shape[0]}")

    # 2. スイング解析
    features, scores = process_landmarks_and_score(landmarks, video_id=video_path)
    
    # 3. 保存
    save_result_json(
        path=output_prefix + ".json",
        video_id=video_file,
        meas=features,
        scores=scores
    )

    save_result_npz(
        path=output_prefix + ".npz",
        landmarks=landmarks,
        features=features,
        scores=scores
    )


    print("解析完了 ✅")
    print(f"特徴量・スコア → {output_prefix}.json")
    print(f"ランドマーク＋結果保存 → {output_prefix}.npz")


# ============ 実行例 ============
if __name__ == "__main__":
    folder_path = "C:/Users/kohei.kuwahara/Desktop/Training/MediaPipe/images/"
    video_file = folder_path + "sample_user.mp4"
    analyze_golf_swing(video_file, output_prefix="sample2_analysis")
