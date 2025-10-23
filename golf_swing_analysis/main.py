from video_io import extract_landmarks_from_video
from swing_scoring import process_landmarks_and_score
from config import ALPHA

def analyze_golf_swing(video_file, output_prefix="outputs/sample_analysis"):
    print(f"動画処理中: {video_file}")

    # ✅ extract_landmarks_from_video は3つの戻り値しか返さない
    landmarks, ball_positions, stationary_frame = extract_landmarks_from_video(video_file)

    # ✅ 特徴量は process_landmarks_and_score 内で計算される
    features, scores = process_landmarks_and_score(
        landmarks,
        video_id=video_file,
        alpha=ALPHA
    )

    # ✅ 結果出力
    print("解析完了 ✅")
    print(f"特徴量・スコア → {output_prefix}.json")
    print(f"ランドマーク＋結果保存 → {output_prefix}.npz")
    print(f"最終スイングスコア: {scores['final_score']:.2f} / 100")
    print(f"パラメータスコア: {scores['param_score']:.3f}")
    print(f"類似度スコア: {scores['similarity_score']:.3f}")

    print("個別項目スコア:")
    for k, v in scores["indiv_scores"].items():
        print(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    video_file = "C:/Users/kohei.kuwahara/Desktop/Training/MediaPipe/images/sample_user1.mp4"
    analyze_golf_swing(video_file)
