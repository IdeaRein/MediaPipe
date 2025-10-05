import json

def print_scores_from_json(json_file: str):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    scores = data.get("scores", {})
    final_score = scores.get("final_score", None)
    param_score = scores.get("param_score", None)
    similarity_score = scores.get("similarity_score", None)
    indiv_scores = scores.get("indiv_scores", {})

    print(f"最終スイングスコア: {final_score:.2f} / 100")
    print(f"パラメータスコア: {param_score:.3f}")
    print(f"類似度スコア: {similarity_score:.3f}")
    print("個別項目スコア:")
    for k, v in indiv_scores.items():
        print(f"  {k}: {v:.3f}")
