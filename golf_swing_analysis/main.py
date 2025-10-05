# main.py
import os
from video_io import extract_landmarks_from_video
from swing_scoring import process_landmarks_and_score, save_result_json, save_result_npz
from sample_analysis import print_scores_from_json

def analyze_golf_swing(video_file, output_prefix="outputs/swing_result"):
    print(f"動画処理中: {video_file}")
    
    # 1. 動画からランドマーク抽出
    landmarks = extract_landmarks_from_video(video_file)
    print(f"抽出フレーム数: {landmarks.shape[0]}")

    # 2. スイング解析
    features, scores = process_landmarks_and_score(landmarks, video_id=video_file)

    # 3. 保存先フォルダ作成
    output_dir = os.path.dirname(output_prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 4. JSON / NPZ 保存
    save_result_json(f"{output_prefix}.json", video_file, features, scores)
    save_result_npz(f"{output_prefix}.npz", landmarks, features, scores)

    print("解析完了 ✅")
    print(f"特徴量・スコア → {output_prefix}.json")
    print(f"ランドマーク＋結果保存 → {output_prefix}.npz")
    
    # 5. 保存した JSON を読み込んでスコア表示
    print_scores_from_json(f"{output_prefix}.json")


if __name__ == "__main__":
    folder = "C:/Users/kohei.kuwahara/Desktop/Training/MediaPipe/images/"
    video_file = os.path.join(folder, "sample_user.mp4")
    #video_file = os.path.join(folder, "sample_matuyamaHideki.mp4")
    #video_file = os.path.join(folder, "sample_TigerWoods.mp4")
    analyze_golf_swing(video_file, output_prefix="outputs/sample2_analysis")
