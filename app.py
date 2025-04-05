import streamlit as st
import numpy as np
import cv2 as cv
import joblib
import os

# Tiêu đề và mô tả cho ứng dụng
st.title('Hệ thống Nhận diện Khuôn mặt')
st.write('Ứng dụng này sử dụng OpenCV để phát hiện và nhận dạng khuôn mặt từ webcam của bạn.')


# Tải các mô hình và cấu hình
@st.cache_resource
def load_models():
    model_folder = "model"

    # Kiểm tra nếu thư mục mô hình tồn tại, tạo nếu không có
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Đường dẫn đến các mô hình
    YUNET_MODEL = os.path.join(model_folder, "face_detection_yunet_2023mar.onnx")
    SFACENET_MODEL = os.path.join(model_folder, "face_recognition_sface_2021dec.onnx")

    # Kiểm tra nếu các mô hình tồn tại
    if not os.path.exists(YUNET_MODEL):
        st.error(f"Face detection model not found at {YUNET_MODEL}. Please download it first.")
        st.markdown("Download from: https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet")
        return None, None, None

    if not os.path.exists(SFACENET_MODEL):
        st.error(f"Face recognition model not found at {SFACENET_MODEL}. Please download it first.")
        st.markdown("Download from: https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface")
        return None, None, None

    # Tải mô hình SVC cho phân loại khuôn mặt
    svc_path = os.path.join(model_folder, "svc.pkl")
    if not os.path.exists(svc_path):
        st.error(f"Không tìm thấy mô hình SVC tại {svc_path}.")
        return None, None, None

    svc = joblib.load(svc_path)

    # Tạo bộ phát hiện và bộ nhận diện
    detector = cv.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),
        0.9,  # ngưỡng điểm số
        0.3,  # ngưỡng nms
        5000  # top_k
    )

    recognizer = cv.FaceRecognizerSF.create(
        SFACENET_MODEL, ""
    )

    return detector, recognizer, svc


# Tải các mô hình cần thiết
detector, recognizer, svc = load_models()

# Danh sách tên cho dự đoán - Đã thay đổi thành tên tiếng Việt
mydict = ['Anh Ngoc', 'Bao', 'Huu Loc', 'Linh Phan', 'Thai Hung']

# Cài đặt thanh bên
with st.sidebar:
    st.header('Cài đặt')
    score_threshold = st.slider('Ngưỡng tin cậy nhận diện khuôn mặt', 0.1, 1.0, 0.9, 0.05)


    recognition_threshold = st.slider('Ngưỡng nhận diện khuôn mặt đã biết', 0.5, )

    # Khởi tạo trạng thái phiên cho nút bật/tắt camera
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False

    # Nút bật/tắt camera
    if st.button('Bật/Tắt Camera'):
        st.session_state.camera_on = not st.session_state.camera_on


# Hàm để hiển thị các khuôn mặt đã phát hiện
def visualize(image, faces, fps, threshold=0.8, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            x, y, w, h = coords[0], coords[1], coords[2], coords[3]

            # Căn chỉnh & trích xuất đặc trưng
            face_align = recognizer.alignCrop(image, face)
            face_feature = recognizer.feature(face_align)

            decision_scores = svc.decision_function(face_feature)
            prediction = svc.predict(face_feature)[0]
            
            confidence = np.max(decision_scores)

            if confidence > threshold:
                label = mydict[prediction]
                color = (0, 255, 0)
            else:
                label = "Unknown"
                color = (0, 0, 255)

            conf_text = f"{confidence:.2f}"

            # Vẽ hộp giới hạn
            cv.rectangle(image, (x, y), (x + w, y + h), color, thickness)

            # Vẽ nhãn tên phía trên hộp
            cv.putText(image, f"{label} ({conf_text})", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Vẽ các điểm đặc trưng
            for i in range(5):
                cx, cy = coords[4 + i * 2], coords[5 + i * 2]
                cv.circle(image, (cx, cy), 2, (255, 0, 255), thickness)

    # Hiển thị FPS
    cv.putText(image, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


# Logic chính của ứng dụng
if detector is not None and recognizer is not None and svc is not None:
    # Tạo placeholder cho luồng webcam
    frame_placeholder = st.empty()

    if st.session_state.camera_on:
        # Mở webcam
        cap = cv.VideoCapture(0)

        if not cap.isOpened():
            st.error("Không thể mở webcam!")
        else:
            # Thiết lập kích thước đầu vào cho bộ phát hiện dựa trên độ phân giải webcam
            frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            detector.setInputSize([frameWidth, frameHeight])

            # Đồng hồ đo FPS
            tm = cv.TickMeter()

            # Hiển thị luồng webcam với nhận diện khuôn mặt
            try:
                while st.session_state.camera_on:
                    tm.start()
                    success, frame = cap.read()
                    if not success:
                        st.error("Không thể đọc từ webcam!")
                        break

                    # Chạy phát hiện khuôn mặt
                    faces = detector.detect(frame)
                    tm.stop()

                    # Xử lý và hiển thị kết quả
                    frame = visualize(frame, faces, tm.getFPS(), recognition_threshold)

                    # Chuyển đổi sang RGB để hiển thị trong Streamlit
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                    # Hiển thị khung hình
                    frame_placeholder.image(frame)

                    # Đặt lại cho khung hình tiếp theo
                    tm.reset()

                    # Cần thiết để Streamlit cập nhật giao diện
                    if not st.session_state.camera_on:
                        break

            finally:
                cap.release()
    else:
        # Hiển thị placeholder khi camera tắt
        st.info('Nhấp "Bật/Tắt Camera" ở sidebar để bắt đầu nhận diện khuôn mặt')
else:
    st.error('Không thể tải các mô hình. Vui lòng kiểm tra tất cả các mô hình cần thiết có sẵn trong thư mục "model".')
