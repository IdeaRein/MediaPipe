import json

json_file = "sample2_analysis.json"

# JSON読み込み
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# スコアを抽出
final_score = data.get("scores", {}).get("final_score", None)
param_score = data.get("scores", {}).get("param_score", None)
similarity_score = data.get("scores", {}).get("similarity_score", None)
indiv_scores = data.get("scores", {}).get("indiv_scores", {})

# 表示
print(f"最終スイングスコア: {final_score:.2f} / 100")
print(f"パラメータスコア: {param_score:.3f}")
print(f"類似度スコア: {similarity_score:.3f}")
print("個別項目スコア:")
for k, v in indiv_scores.items():
    print(f"  {k}: {v:.3f}")
