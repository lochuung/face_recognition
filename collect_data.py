import cv2
import os
import numpy as np
import time

YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

def collect_face_samples(name, num_samples=10):
    # create folder
    folder_path = f"face_dataset/{name}"
    os.makedirs(folder_path, exist_ok=True)

    # camera
    cap = cv2.VideoCapture(0)

    # face detector
    face_detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL_PATH,
        "",
        (320, 320),
        score_threshold=0.9,
        nms_threshold=0.3
    )

    count = 0

    print(f"Bắt đầu thu thập dữ liệu cho {name}")
    print("Hãy nhìn thẳng vào camera và di chuyển đầu chậm để có nhiều góc khác nhau")

    # sleep 3 seconds
    time.sleep(3)

    while count < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi khi đọc frame từ camera")
            break

        # frame size
        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        # detect face
        _, faces = face_detector.detect(frame)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            x, y, w, h, conf = face[0:5].astype(np.int32)

            # check confidence
            if conf > 0.9:
                face_img = frame[y: y + h, x: x + w]

                # save face image
                file_path = f"{folder_path}/image_{count + 1:04d}.jpg"
                cv2.imwrite(file_path, face_img)

                count += 1
                print(f"Đã thu thập {count}/{num_samples} mẫu")

                # show notification
                cv2.putText(frame, f"Đã thu thập {count}/{num_samples} mẫu",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                time.sleep(0.5)

            # draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show frame
        cv2.imshow("Face Collection", frame)

        # press q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release camera
    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập xong dữ liệu cho {name}")
    return folder_path

collect_face_samples("Huu Loc", num_samples=10)
