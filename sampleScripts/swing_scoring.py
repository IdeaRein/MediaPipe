# swing_scoring.py
import numpy as np
import math
import json
from typing import List, Dict, Any, Tuple

# ---------- ユーティリティ ----------
def angle_3pt(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])  # 2Dで計算
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosv = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

def line_angle(p1, p2):
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    return float(np.degrees(math.atan2(dy, dx)))

def smooth_series(x: np.ndarray, window: int = 5) -> np.ndarray:
    if x.ndim == 1:
        pad = window//2
        x_p = np.pad(x, pad, mode='edge')
        kernel = np.ones(window) / window
        return np.convolve(x_p, kernel, mode='valid')
    else:
        # axis=0 に沿って平滑化
        out = np.zeros_like(x)
        for i in range(x.shape[1]):
            out[:, i] = smooth_series(x[:, i], window)
        return out

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten(); b = b.flatten()
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------- 前処理：ランドマーク配列を正規化 ----------
def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks: shape (T, 33, 4) with x,y,z,vis (x,y normalized 0..1)
    正規化: 各フレームを肩幅でスケールし、センタリング(骨盤中心)する
    """
    T = landmarks.shape[0]
    out = landmarks.copy()
    for t in range(T):
        L = out[t]
        # 使用する index は MediaPipe の標準に準拠
        left_sh = L[11][:2]; right_sh = L[12][:2]
        left_hip = L[23][:2]; right_hip = L[24][:2]
        # 肩幅
        shoulder_w = np.linalg.norm(np.array(right_sh) - np.array(left_sh)) + 1e-6
        # 骨盤中心を原点にする
        pelvis = (np.array(left_hip) + np.array(right_hip)) / 2.0
        # x,y を pelvis 基準に移動して肩幅で割る
        out[t, :, 0] = (out[t, :, 0] - pelvis[0]) / shoulder_w
        out[t, :, 1] = (out[t, :, 1] - pelvis[1]) / shoulder_w
        # z はそのまま（正規化したスケールで相対値になる）
    return out

# ---------- キーフレーム検出（簡易） ----------
def detect_keyframes(landmarks: np.ndarray, fps: int = 30) -> Dict[str, int]:
    """
    シンプルなキーフレーム検出（バックスイングトップ、インパクト推定）
    - top: 手首（右/左どちらか速さが0に近い頂点）を参照（ここは heuristic）
    - impact: 手の速度が最大になる直前のピークをインパクト候補とする
    """
    T = landmarks.shape[0]
    # 手首: left=15 right=16 (MediaPipe の index)
    left_wr = landmarks[:, 15, :2]
    right_wr = landmarks[:, 16, :2]
    # 使用するのは利き手想定？ここは右利き/左利きの考慮が必要。暫定：左右合成速度を利用
    pos = (left_wr + right_wr) / 2.0
    # 速度（フレーム差）
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    # smoothed
    vel_s = smooth_series(vel, window=7)
    if len(vel_s) == 0:
        return {"top": 0, "impact": T-1}
    # impact: 速度がピークとなるフレームindex（+1 due to diff）
    impact_idx = int(np.argmax(vel_s) + 1)
    # top: 前半で手の移動が最小となる箇所（単純）
    half = max(1, T//2)
    top_region = pos[:half]
    # 手のy（上方向）が最大（最上点）を top とする（y座標は正規化済み、上下は座標系に依存）
    # note: より良いtop検出は肩角の変化や手首速度の零交差を用いる
    top_idx = int(np.argmax(top_region[:,1]))
    return {"top": top_idx, "impact": impact_idx}

# ---------- 指標抽出 ----------
def extract_summary_features(landmarks: np.ndarray, fps: int = 30) -> Dict[str, Any]:
    """
    landmarks normalized => shape (T, 33, 4)
    return: dict of numeric features (shoulder_top, hip_top, elbow_at_impact, tempo_ratio, ...)
    """
    T = landmarks.shape[0]
    kf = detect_keyframes(landmarks, fps)
    top = kf["top"]; impact = kf["impact"]

    # helper to get (x,y) tuple
    def L(t, idx): return landmarks[t, idx, :2]

    # shoulder angle (line angle between left_shoulder and right_shoulder) at top
    shoulder_angle_top = line_angle(L(top, 11), L(top, 12))
    hip_angle_top = line_angle(L(top, 23), L(top, 24))
    # X-factor: shoulder - hip
    x_factor = shoulder_angle_top - hip_angle_top

    # elbow angle at impact (choose lead arm; assume right-handed user -> left arm is lead? -> let's compute both and choose the one closer to 180)
    left_elb = angle_3pt(L(impact, 11), L(impact, 13), L(impact, 15))
    right_elb = angle_3pt(L(impact, 12), L(impact, 14), L(impact, 16))
    # choose lead as the one more extended (closer to 180)
    elbow_at_impact = left_elb if abs(180-left_elb) < abs(180-right_elb) else right_elb

    # tempo: (time from address to top) / (time from top to impact)
    # approximate address as frame 0
    t_back = max(1, top) / fps
    t_down = max(1, impact - top) / fps
    tempo_ratio = (t_back / t_down) if t_down>0 else float('inf')

    # head stability: std of nose y across swing (lower is better)
    nose_y_series = landmarks[:, 0, 1]
    head_std = float(np.std(nose_y_series))

    # hand path deviation surrogate: fit line to wrist trajectory (avg of both) and compute mean orthogonal distance
    left_wr = landmarks[:, 15, :2]
    right_wr = landmarks[:, 16, :2]
    hand_pos = (left_wr + right_wr) / 2.0
    # simple linear regression line (x as independent)
    xs = hand_pos[:,0]; ys = hand_pos[:,1]
    if np.ptp(xs) < 1e-6:
        hand_dev = 0.0
    else:
        p = np.polyfit(xs, ys, 1)
        ys_fit = np.polyval(p, xs)
        hand_dev = float(np.mean(np.abs(ys - ys_fit)))

    # assemble
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

# ---------- 正規化関数（目標値 + tolerance を使う） ----------
def score_from_deviation(value: float, ideal: float, tol: float) -> float:
    """
    value: measured
    ideal: target value
    tol: allowed deviation (90%以内くらいを1.0にする尺度)
    returns 0..1 (clipped)
    """
    diff = abs(value - ideal)
    s = max(0.0, 1.0 - (diff / (tol + 1e-9)))
    return s

def score_tempo_ratio(ratio: float, ideal: float = 3.0, tol: float = 1.0) -> float:
    # tempo is ratio so asymmetric tolerance may be needed; use relative diff
    diff = abs(ratio - ideal)
    s = max(0.0, 1.0 - (diff / (tol + 1e-9)))
    return s

# ---------- ハイブリッド合成 ----------
DEFAULT_WEIGHTS = {
    "shoulder": 0.20,
    "hip": 0.20,
    "x_factor": 0.12,
    "elbow": 0.15,
    "tempo": 0.18,
    "hand_path": 0.10,
    "head": 0.05
}

def compute_scores(meas: Dict[str, Any], pro_reference: Dict[str, float] = None,
                   weights: Dict[str, float] = None, alpha: float = 0.7) -> Dict[str, Any]:
    """
    meas: features dict from extract_summary_features
    pro_reference: dict of ideal/pro avg values; if None, use some reasonable defaults
    weights: per-parameter weights (sum should be 1.0 ideally)
    alpha: weight for param-based score vs vector similarity (param_score * alpha + sim * (1-alpha))
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    if pro_reference is None:
        pro_reference = {
            "shoulder_angle_top": 172.59,  # degree (example)
            "hip_angle_top": 174.85,
            "x_factor_top": -2.25,
            "elbow_at_impact": 53.34,
            "tempo_ratio": 0.78,
            "hand_path_dev": 3.88,  # normalized units (shoulder-width scale)
            "head_std": 6.49
        }

    # individual normalized scores 0..1
    s_shoulder = score_from_deviation(meas["shoulder_angle_top"], pro_reference["shoulder_angle_top"], tol=20.0)
    s_hip = score_from_deviation(meas["hip_angle_top"], pro_reference["hip_angle_top"], tol=20.0)
    s_x = score_from_deviation(meas["x_factor_top"], pro_reference["x_factor_top"], tol=15.0)
    s_elbow = score_from_deviation(meas["elbow_at_impact"], pro_reference["elbow_at_impact"], tol=10.0)
    s_tempo = score_tempo_ratio(meas["tempo_ratio"], pro_reference["tempo_ratio"], tol=1.5)
    s_hand = score_from_deviation(meas["hand_path_dev"], pro_reference["hand_path_dev"], tol=0.05)
    s_head = score_from_deviation(meas["head_std"], pro_reference["head_std"], tol=0.02)

    indiv_scores = {
        "shoulder": s_shoulder,
        "hip": s_hip,
        "x_factor": s_x,
        "elbow": s_elbow,
        "tempo": s_tempo,
        "hand_path": s_hand,
        "head": s_head
    }

    # weighted param_score
    total_weight = sum(weights.values())
    param_score = 0.0
    for k, w in weights.items():
        param_score += indiv_scores.get(k, 0.0) * (w / total_weight)

    # vector similarity: make embeddings from a selection of normalized features
    # create normalized feature vector in a fixed order
    vec_meas = np.array([
        meas["shoulder_angle_top"] / 180.0,
        meas["hip_angle_top"] / 180.0,
        meas["x_factor_top"] / 180.0,
        meas["elbow_at_impact"] / 180.0,
        (meas["tempo_ratio"] / 5.0),  # assuming ratio range ~0..5
        meas["hand_path_dev"] / 0.2,
        meas["head_std"] / 0.1
    ], dtype=float)

    # pro vector
    vec_pro = np.array([
        pro_reference["shoulder_angle_top"] / 180.0,
        pro_reference["hip_angle_top"] / 180.0,
        pro_reference["x_factor_top"] / 180.0,
        pro_reference["elbow_at_impact"] / 180.0,
        pro_reference["tempo_ratio"] / 5.0,
        pro_reference["hand_path_dev"] / 0.2,
        pro_reference["head_std"] / 0.1
    ], dtype=float)

    sim = cosine_similarity(vec_meas, vec_pro)  # 0..1 (maybe negative if opp; but should be >=0 here)
    sim = max(0.0, sim)  # clip

    # final hybrid
    final_raw = alpha * param_score + (1.0 - alpha) * sim
    final_score_0_100 = float(np.clip(final_raw, 0.0, 1.0) * 100.0)

    # confidence: based on visibility and frames coverage (simple heuristic)
    # if many landmarks have low visibility, lower confidence
    # Here we don't have full vis array passed; user can pass if desired. We'll set a placeholder 0.9
    confidence = 0.9

    return {
        "indiv_scores": indiv_scores,
        "param_score": float(param_score),
        "similarity_score": float(sim),
        "final_score": final_score_0_100,
        "confidence": float(confidence),
        "vec_meas": vec_meas.tolist(),
        "vec_pro": vec_pro.tolist()
    }

# ---------- 保存 ----------
def save_result_json(path: str, video_id: str, meas: Dict[str, Any], scores: Dict[str, Any]):
    out = {
        "video_id": video_id,
        "frames": meas.get("frames"),
        "top_frame": meas.get("top_frame"),
        "impact_frame": meas.get("impact_frame"),
        "features": meas,
        "scores": scores
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

def save_result_npz(path: str, landmarks: np.ndarray, features: Dict[str, Any], scores: Dict[str, Any]):
    np.savez_compressed(path, landmarks=landmarks, features=features, scores=scores)

# ---------- フルパイプラインユーティリティ ----------
def process_landmarks_and_score(landmarks_raw: np.ndarray, video_id: str = "video",
                                fps: int = 30, pro_reference: Dict[str,float] = None,
                                weights: Dict[str,float] = None, alpha: float = 0.7):
    """
    landmarks_raw: (T, 33, 4) with x,y,z,vis normalized 0..1
    returns scores dict and optionally saves files
    """
    # 1. 前処理（正規化 + スムージング）
    lm_norm = normalize_landmarks(landmarks_raw)
    # smooth x,y series
    lm_xy = lm_norm[:, :, :2]
    lm_xy_sm = smooth_series(lm_xy, window=5)
    lm_norm[:, :, :2] = lm_xy_sm

    # 2. 特徴量抽出
    features = extract_summary_features(lm_norm, fps=fps)

    # 3. スコア計算
    scores = compute_scores(features, pro_reference=pro_reference, weights=weights, alpha=alpha)
    return features, scores

# Example usage:
if __name__ == "__main__":
    # ダミーデータの作成例：T=120 フレーム, 各landmark x,y in [0..1], z=0, vis=1
    T = 120
    landmarks_dummy = np.random.rand(T, 33, 4)
    landmarks_dummy[:, :, 2] = 0.0
    landmarks_dummy[:, :, 3] = 1.0

    features, scores = process_landmarks_and_score(landmarks_dummy, video_id="sample")
    print("features:", features)
    print("scores:", scores)
