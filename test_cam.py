import cv2

for i in range(3):  # Vì bạn có 3 camera đang hoạt động
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Hiển thị camera index {i} - Nhấn ESC để chuyển tiếp.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f'Camera {i}', frame)
            if cv2.waitKey(1) == 27:  # ESC để đóng
                break
        cap.release()
        cv2.destroyAllWindows()
