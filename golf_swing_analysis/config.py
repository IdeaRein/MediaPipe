# config.py
# ゴルフスイング解析の設定値（理論値）
PRO_REFERENCE = {
    "shoulder_angle_top": 85.0,   # 目標：実肩回転 ≈ 90°。DTL投影で若干小さく見えるので85°を基準に。
    "hip_angle_top": 45.0,        # 実腰回転 ≈ 45〜50°。DTL投影で小さめに見えるが45°を基準に。
    "x_factor_top": 40.0,         # 肩−腰差は40〜50°が目安。投影で縮むため40°基準。
    "elbow_at_impact": 145.0,     # インパクトは140〜150°（ほぼ伸び）を理想。
    "tempo_ratio": 3.0,           # バックスイング：ダウンスイング ≈ 3:1。
    "hand_path_dev": 0.32,        # 正規化値。DTLでは軌道の広がりが見えやすいので0.25〜0.35の中間。
    "head_std": 2.0               # ピクセル/正規化単位で2.0以下を「安定」判定。
}

# 松山英樹のスイングデータを参考にした理想値(後方動画の場合)
# PRO_REFERENCE = {
#     "shoulder_angle_top": -175.27,
#     "hip_angle_top": 175.26,
#     "x_factor_top": -350.53,
#     "elbow_at_impact": 116.88,
#     "tempo_ratio": 0.01,
#     "hand_path_dev": 0.62,
#     "head_std": 2.94
# }


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
