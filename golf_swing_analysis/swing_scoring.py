# swing_scoring.py
import numpy as np
import json
from swing_features import normalize_landmarks, extract_summary_features
from utils import angle_3pt, line_angle, smooth_series, cosine_similarity
from config import PRO_REFERENCE, DEFAULT_WEIGHTS, ALPHA

def score_from_deviation(value, ideal, tol):
    diff = abs(value-ideal)
    return max(0.0, 1.0 - diff/(tol+1e-9))

def score_tempo_ratio(ratio, ideal=3.0, tol=1.0):
    diff = abs(ratio-ideal)
    return max(0.0, 1.0 - diff/(tol+1e-9))

def compute_scores(meas, pro_reference=None, weights=None, alpha=ALPHA):
    if pro_reference is None: pro_reference = PRO_REFERENCE
    if weights is None: weights = DEFAULT_WEIGHTS

    indiv_scores = {
        "shoulder": score_from_deviation(meas["shoulder_angle_top"], pro_reference["shoulder_angle_top"], 20),
        "hip": score_from_deviation(meas["hip_angle_top"], pro_reference["hip_angle_top"], 20),
        "x_factor": score_from_deviation(meas["x_factor_top"], pro_reference["x_factor_top"], 15),
        "elbow": score_from_deviation(meas["elbow_at_impact"], pro_reference["elbow_at_impact"], 10),
        "tempo": score_tempo_ratio(meas["tempo_ratio"], pro_reference["tempo_ratio"], 1.5),
        "hand_path": score_from_deviation(meas["hand_path_dev"], pro_reference["hand_path_dev"], 0.05),
        "head": score_from_deviation(meas["head_std"], pro_reference["head_std"], 0.02)
    }

    total_weight = sum(weights.values())
    param_score = sum(indiv_scores[k]*weights[k]/total_weight for k in weights)

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
    final_score = float(np.clip(alpha*param_score + (1-alpha)*sim,0,1)*100)

    return {
        "indiv_scores": indiv_scores,
        "param_score": float(param_score),
        "similarity_score": float(sim),
        "final_score": final_score,
        "confidence": 0.9,
        "vec_meas": vec_meas.tolist(),
        "vec_pro": vec_pro.tolist()
    }

def save_result_json(path, video_id, meas, scores):
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

def save_result_npz(path, landmarks, features, scores):
    import numpy as np
    np.savez_compressed(path, landmarks=landmarks, features=features, scores=scores)

def process_landmarks_and_score(landmarks_raw, video_id="video", fps=30,
                                pro_reference=None, weights=None, alpha=ALPHA):
    lm_norm = normalize_landmarks(landmarks_raw)
    lm_xy = lm_norm[:, :, :2]
    lm_norm[:, :, :2] = smooth_series(lm_xy, window=5)
    features = extract_summary_features(lm_norm, fps=fps)
    scores = compute_scores(features, pro_reference=pro_reference, weights=weights, alpha=alpha)
    return features, scores
