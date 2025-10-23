import numpy as np
import json
from swing_features import normalize_landmarks, extract_summary_features
from utils import cosine_similarity, smooth_series
from config import PRO_REFERENCE, DEFAULT_WEIGHTS, ALPHA


def compute_scores(meas, pro_reference=None, weights=None, alpha=ALPHA):
    if pro_reference is None: pro_reference = PRO_REFERENCE
    if weights is None: weights = DEFAULT_WEIGHTS

    indiv_scores = {
        "shoulder": abs(1 - abs(meas["shoulder_angle_top"] - pro_reference["shoulder_angle_top"]) / 20),
        "hip": abs(1 - abs(meas["hip_angle_top"] - pro_reference["hip_angle_top"]) / 20),
        "x_factor": abs(1 - abs(meas["x_factor_top"] - pro_reference["x_factor_top"]) / 15),
        "elbow": abs(1 - abs(meas["elbow_at_impact"] - pro_reference["elbow_at_impact"]) / 10),
        "tempo": abs(1 - abs(meas["tempo_ratio"] - pro_reference["tempo_ratio"]) / 1.5),
        "hand_path": abs(1 - abs(meas["hand_path_dev"] - pro_reference["hand_path_dev"]) / 0.05),
        "head": abs(1 - abs(meas["head_std"] - pro_reference["head_std"]) / 0.02)
    }

    total_weight = sum(weights.values())
    param_score = sum(indiv_scores[k] * weights[k] / total_weight for k in weights)

    vec_meas = np.array([
        meas["shoulder_angle_top"]/180, meas["hip_angle_top"]/180,
        meas["x_factor_top"]/180, meas["elbow_at_impact"]/180,
        meas["tempo_ratio"]/5, meas["hand_path_dev"]/0.2, meas["head_std"]/0.1
    ])
    vec_pro = np.array([
        pro_reference["shoulder_angle_top"]/180, pro_reference["hip_angle_top"]/180,
        pro_reference["x_factor_top"]/180, pro_reference["elbow_at_impact"]/180,
        pro_reference["tempo_ratio"]/5, pro_reference["hand_path_dev"]/0.2, pro_reference["head_std"]/0.1
    ])
    sim = max(0.0, cosine_similarity(vec_meas, vec_pro))
    final_score = float(np.clip(alpha * param_score + (1 - alpha) * sim, 0, 1) * 100)

    return {
        "indiv_scores": indiv_scores,
        "param_score": float(param_score),
        "similarity_score": float(sim),
        "final_score": final_score
    }


def process_landmarks_and_score(landmarks_raw, video_id="video", fps=30,
                                pro_reference=None, weights=None, alpha=ALPHA,
                                override_features=None):
    lm_norm = normalize_landmarks(landmarks_raw)
    lm_xy = lm_norm[:, :, :2]
    lm_norm[:, :, :2] = smooth_series(lm_xy, window=5)

    # ✅ override_features があればそれを使用（impact_frame_real含む）
    if override_features is not None:
        features = override_features
    else:
        features = extract_summary_features(lm_norm, fps=fps, video_path=video_id)

    scores = compute_scores(features, pro_reference=pro_reference, weights=weights, alpha=alpha)
    return features, scores
