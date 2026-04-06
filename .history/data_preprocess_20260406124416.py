import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

img_size = 64
sequence_length = 16
dataset_dir = "Dataset"

class_map = {
    "Violence": 1,
    "Nonviolence": 0
}

def extract_frames_from_video(video_path, sequence_length=16, img_size=64):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    skip = max(total_frames // sequence_length, 1)
    frames = []

    for i in range(sequence_length):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip)
        success, frame = cap.read()

        if not success or frame is None:
            break

        frame = cv2.resize(frame, (img_size, img_size))
        frame = frame / 255.0
        frames.append(frame)

    cap.release()
    return frames

def create_dataset():
    features = []
    labels = []

    valid_ext = (".mp4", ".avi", ".mov", ".mkv")

    for class_name, class_label in class_map.items():
        class_folder = os.path.join(dataset_dir, class_name)

        if not os.path.exists(class_folder):
            print(f"Không tìm thấy folder: {class_folder}")
            continue

        print("Processing:", class_name)

        for file_name in os.listdir(class_folder):
            if not file_name.lower().endswith(valid_ext):
                continue

            video_path = os.path.join(class_folder, file_name)
            frames = extract_frames_from_video(video_path, sequence_length, img_size)

            if len(frames) == sequence_length:
                features.append(frames)
                labels.append(class_label)
            else:
                print("Bỏ qua video không đủ frame:", video_path)

    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return features, labels

features, labels = create_dataset()

print("Features shape:", features.shape)
print("Labels shape:", labels.shape)