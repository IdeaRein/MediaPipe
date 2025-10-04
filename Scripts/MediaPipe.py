import cv2
import mediapipe as mp

# Mediapipeの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 動画キャプチャの初期化
images_folder_path = "C:/Users/kohei.kuwahara/Desktop/Training/MediaPipe/images/"
cap = cv2.VideoCapture(images_folder_path + "sample2.mp4")


if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
resize_scale = 0.5

# 保存する動画の設定
output_filename = "output_pose_video.avi"
fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # 縮小後の幅
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # 縮小後の高さ
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画のコーデック

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: フレームを取得できませんでした。")
        break

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

    # フレームサイズを縮小
    small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))

    # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Mediapipeで骨格検出を実行
    result = pose.process(frame_rgb)

    # 検出結果を描画
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            small_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        # 各関節の座標を取得して出力
        for i, landmark in enumerate(result.pose_landmarks.landmark):
            x = landmark.x * width    # x座標をピクセル単位に変換
            y = landmark.y * height   # y座標をピクセル単位に変換
            z = landmark.z            # z座標（深度情報）は正規化されている
            print(f"関節 {i}: x={x:.2f}, y={y:.2f}, z={z:.2f}")

    # 縮小されたフレームを保存
    out.write(small_frame)

    # 縮小されたフレームを表示
    cv2.imshow('Pose Detection', small_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()  # 保存用のVideoWriterを解放
cv2.destroyAllWindows()
print(f"保存された動画ファイル: {output_filename}")

