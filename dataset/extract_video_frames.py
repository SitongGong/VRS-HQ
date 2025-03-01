import cv2
import os

def extract_frames(video_path, output_folder, fps=30):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    # 获取视频的帧率
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, video_fps // fps)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取完成

        # 仅保存指定间隔的帧
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    print(f"Finished: {saved_frame_count} frames saved to {output_folder}")

# 使用示例
input_video_file = "/18515601223/Videos"
video_name_list = os.listdir(input_video_file)
for video_name in video_name_list:
    video_root = os.path.join(input_video_file, video_name)
    output_dir = os.path.join('/18515601223/segment-anything-2/extract_frames', video_name.split('.')[0])
    extract_frames(video_root, output_dir, fps=1)
