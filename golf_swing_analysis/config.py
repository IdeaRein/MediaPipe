# config.py
PRO_REFERENCE = {
    "shoulder_angle_top": 172.59,
    "hip_angle_top": 174.85,
    "x_factor_top": -2.25,
    "elbow_at_impact": 53.34,
    "tempo_ratio": 0.78,
    "hand_path_dev": 3.88,
    "head_std": 6.49
}

DEFAULT_WEIGHTS = {
    "shoulder": 0.20,
    "hip": 0.20,
    "x_factor": 0.12,
    "elbow": 0.15,
    "tempo": 0.18,
    "hand_path": 0.10,
    "head": 0.05
}

ALPHA = 0.7  # パラメータスコアと類似度スコアのハイブリッド比率
