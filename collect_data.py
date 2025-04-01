import cv2
import os
import numpy as np
import time

YUNET_MODEL_PATH = "model/face_detection_yunet_2023mar.onnx"
FACE_IMG_SIZE = (112, 112)  # Kích thước chuẩn để resize face
DEFAULT_CAMERA_ID = 0


def init_face_detector(score_threshold=0.9, nms_threshold=0.3):
    return cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        score_threshold,
        nms_threshold
    )


def save_face(face_img, folder_path, count):
    face_img_resized = cv2.resize(face_img, FACE_IMG_SIZE)
    file_path = os.path.join(folder_path, f"image_{count:04d}.jpg")
    cv2.imwrite(file_path, face_img_resized)


def draw_feedback(frame, face, count, total):
    x, y, w, h, _ = face[0:5].astype(np.int32)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(frame, f"Captured {count}/{total}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def collect_face_samples(name, num_samples=120, camera_id=DEFAULT_CAMERA_ID):
    folder_path = f"face_dataset/{name}"
    os.makedirs(folder_path, exist_ok=True)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise IOError("Không mở được camera")

    face_detector = init_face_detector()
    print(f"Bắt đầu thu thập dữ liệu cho '{name}'...")
    print("Hãy nhìn thẳng vào camera và di chuyển nhẹ nhàng đầu.")

    time.sleep(2)
    count = 0
    prev_time = time.time()

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi khi đọc frame từ camera")
            break

        height, width = frame.shape[:2]
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(frame)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            x, y, w, h, conf = face[0:5].astype(np.int32)

            if conf > 0.9:
                face_img = frame[y:y + h, x:x + w]
                if face_img.size > 0:
                    save_face(face_img, folder_path, count + 1)
                    draw_feedback(frame, face, count + 1, num_samples)
                    count += 1

        # Tính và hiển thị FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Face Collection", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập {count} ảnh cho '{name}' tại '{folder_path}'")
    return folder_path


if __name__ == "__main__":
    collect_face_samples("Anh Ngoc")
