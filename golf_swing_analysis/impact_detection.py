import os
import numpy as np
from utils import extract_audio, detect_audio_impact
from swing_features import refine_impact_by_motion
from video_io import detect_ball_stationary_frame


def detect_hybrid_impact(video_path: str, fps: int = 30, output_dir: str = "outputs"):
    """
    音＋骨格＋ボール静止のハイブリッドでインパクトを検出
    """
    os.makedirs(output_dir, exist_ok=True)

    # === Step1: 音声を抽出してオンセットを検出 ===
    audio_path = extract_audio(video_path)
    audio_frame = detect_audio_impact(audio_path, fps)
    search_window = (max(0, (audio_frame or 0) - 5), (audio_frame or 0) + 5)

    # === Step2: ボール静止フレームの検出 ===
    ball_path = os.path.join(output_dir, "ball_positions.npy")
    if not os.path.exists(ball_path):
        raise FileNotFoundError(f"ボール座標ファイルが見つかりません: {ball_path}\n先に video_io.extract_landmarks_from_video() を実行してください。")
    ball_positions = np.load(ball_path)

    stationary_frame = detect_ball_stationary_frame(ball_positions, threshold=1.0, stable_count=5)

    # === Step3: 骨格モーション解析 ===
    lm_path = os.path.join(output_dir, "landmarks.npy")
    if not os.path.exists(lm_path):
        raise FileNotFoundError(f"ランドマークファイルが見つかりません: {lm_path}\n先に video_io.extract_landmarks_from_video() を実行してください。")
    landmarks = np.load(lm_path)

    # ✅ 修正1: 音がNoneの場合も考慮して fallback
    if audio_frame is None and stationary_frame is None:
        rough_idx = 0
    elif audio_frame is None:
        rough_idx = stationary_frame
    elif stationary_frame is None:
        rough_idx = audio_frame
    else:
        # 音とボール静止が両方あれば「先に起こる方」を優先
        rough_idx = min(audio_frame, stationary_frame)

    refined_idx = refine_impact_by_motion(landmarks, fps, rough_idx, window=3)

    # === Step4: 統合判定 ===
    if stationary_frame is not None:
        impact_frame = min(refined_idx, stationary_frame)
        confidence = 0.95
    elif audio_frame is not None:
        impact_frame = refined_idx
        confidence = 0.85
    else:
        impact_frame = refined_idx
        confidence = 0.7

    # === Step5: ログ出力 ===
    print("========== インパクト検出結果 ==========")
    print(f"🎧 音声推定フレーム: {audio_frame}")
    print(f"⚪ ボール静止フレーム: {stationary_frame}")
    print(f"🏌️ 精密補正後フレーム: {refined_idx}")
    print(f"✅ 最終インパクトフレーム: {impact_frame}")
    print(f"📊 信頼度: {confidence:.2f}")
    print("=======================================")

    return {
        "impact_frame": int(impact_frame),
        "audio_frame": int(audio_frame) if audio_frame is not None else None,
        "stationary_frame": int(stationary_frame) if stationary_frame is not None else None,
        "confidence": confidence,
    }
