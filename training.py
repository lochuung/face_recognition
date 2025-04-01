import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

# === Model paths ===
YUNET_MODEL = "model/face_detection_yunet_2023mar.onnx"
SFACENET_MODEL = "model/face_recognition_sface_2021dec.onnx"
DATASET_DIR = "face_dataset"
MODEL_OUT_PATH = "model/svc.pkl"


# === Dataset class ===
class IdentityMetadata:
    def __init__(self, base, name, file):
        self.base = base
        self.name = name
        self.file = file

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

    def __repr__(self):
        return self.image_path()


def load_metadata(path):
    metadata = []
    for person_name in sorted(os.listdir(path)):
        person_path = os.path.join(path, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_file in sorted(os.listdir(person_path)):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.bmp')):
                metadata.append(IdentityMetadata(path, person_name, img_file))
    return np.array(metadata)


# === Load and align image ===
def align_face(img, detector, recognizer):
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is not None and len(faces) > 0:
        aligned = recognizer.alignCrop(img, faces[0])
        return aligned
    return None


# === Main logic ===
if __name__ == "__main__":
    metadata = load_metadata(DATASET_DIR)
    embedded = np.zeros((metadata.shape[0], 128))

    # Init detector and recognizer
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(SFACENET_MODEL, "")

    used_indices = []
    for i, m in enumerate(metadata):
        print(f"Processing: {m.image_path()}")
        img = cv2.imread(m.image_path())
        aligned = align_face(img, detector, recognizer)
        if aligned is not None:
            feature = recognizer.feature(aligned)
            embedded[i] = feature
            used_indices.append(i)
        else:
            print(f"Không tìm thấy khuôn mặt trong ảnh: {m.image_path()}")

    embedded = embedded[used_indices]
    metadata = metadata[used_indices]

    targets = np.array([m.name for m in metadata])
    encoder = LabelEncoder()
    y = encoder.fit_transform(targets)

    # Tách train/test theo index
    indices = np.arange(metadata.shape[0])
    train_idx = indices % 5 != 0
    test_idx = indices % 5 == 0

    X_train = embedded[train_idx]
    y_train = y[train_idx]
    X_test = embedded[test_idx]
    y_test = y[test_idx]

    # Train model
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    acc = accuracy_score(y_test, svc.predict(X_test))
    print(f"SVM Accuracy: {acc:.4f}")

    joblib.dump(svc, MODEL_OUT_PATH)
    print(f"Saved trained model to: {MODEL_OUT_PATH}")
