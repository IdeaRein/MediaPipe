import numpy as np
import math
import os
import subprocess
import librosa


# =========================================================
# 🎯 幾何・スイング解析用ユーティリティ
# =========================================================
def angle_3pt(a, b, c):
    """3点の角度（度）を返す"""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    cosv = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))


def line_angle(p1, p2):
    """2点を結ぶ線の角度（度）を返す"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(math.atan2(dy, dx)))


def smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    """移動平均で系列をスムージング"""
    if x.ndim == 1:
        pad = window // 2
        x_p = np.pad(x, pad, mode='edge')
        kernel = np.ones(window) / window
        return np.convolve(x_p, kernel, mode='valid')
    else:
        out = np.zeros_like(x)
        for i in range(x.shape[1]):
            out[:, i] = smooth_series(x[:, i], window)
        return out


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """2ベクトル間のコサイン類似度"""
    a, b = a.flatten(), b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# =========================================================
# 🎧 音声処理ユーティリティ
# =========================================================
def extract_audio(video_path: str) -> str:
    """
    動画ファイルから音声（WAV）を抽出して保存
    ffmpeg がインストールされている必要あり。
    """
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",  # 映像を無視
        "-ac", "1",  # モノラル
        "-ar", "44100",  # サンプリングレート
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def detect_audio_impact(audio_path: str, fps: int) -> int | None:
    """
    音のオンセット（急激な音の立ち上がり）を検出し、
    インパクト候補フレームを返す。
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True, units='frames')
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)

        if len(onset_times) == 0:
            return None

        # 最初の強いオンセットをインパクト候補とする
        return int(round(onset_times[0] * fps))

    except Exception as e:
        print("⚠️ detect_audio_impact エラー:", e)
        return None
