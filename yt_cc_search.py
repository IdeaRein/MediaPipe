import requests
import json
import csv
import time

# === 設定項目 ===
API_KEY = "AIzaSyBRA0icTwVwFLtqDLWeLF3aIF8XfjwvmrY"  # ←ここにAPIキーを入れる
QUERY = "Pro golf swing"          # 検索キーワード
MAX_RESULTS = 20              # 1回の取得件数（最大50）
OUTPUT_JSON = "cc_videos.json"
OUTPUT_CSV = "cc_videos.csv"

# === APIエンドポイント ===
SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# === ステップ1：Creative Commons動画を検索 ===
params = {
    "part": "snippet",
    "q": QUERY,
    "type": "video",
    "videoLicense": "creativeCommon",
    "maxResults": MAX_RESULTS,
    "key": API_KEY,
}

print(f"🔍 Searching YouTube Creative Commons videos for: '{QUERY}' ...")
res = requests.get(SEARCH_URL, params=params)
res.raise_for_status()
data = res.json()

video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item["id"]]
print(f"✅ Found {len(video_ids)} videos.")

# === ステップ2：各動画の詳細情報を取得 ===
details_params = {
    "part": "snippet,contentDetails,status",
    "id": ",".join(video_ids),
    "key": API_KEY,
}
details_res = requests.get(VIDEOS_URL, params=details_params)
details_res.raise_for_status()
details_data = details_res.json()

# === ステップ3：結果整形 ===
videos_info = []
for v in details_data.get("items", []):
    vid = v["id"]
    snippet = v["snippet"]
    status = v.get("status", {})

    info = {
        "videoId": vid,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "title": snippet.get("title"),
        "channelTitle": snippet.get("channelTitle"),
        "publishedAt": snippet.get("publishedAt"),
        "description": snippet.get("description"),
        "license": status.get("license", "unknown"),
        "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
    }
    videos_info.append(info)

# === ステップ4：JSON形式で保存 ===
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(videos_info, f, ensure_ascii=False, indent=2)
print(f"💾 JSON saved to: {OUTPUT_JSON}")

# === ステップ5：CSV形式で保存 ===
csv_fields = [
    "videoId", "url", "title", "channelTitle",
    "publishedAt", "license", "thumbnail"
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    for v in videos_info:
        writer.writerow({k: v.get(k, "") for k in csv_fields})

print(f"💾 CSV saved to: {OUTPUT_CSV}")

# === ステップ6：コンソール表示 ===
for v in videos_info:
    print(f"{v['title']} ({v['license']}) → {v['url']}")
