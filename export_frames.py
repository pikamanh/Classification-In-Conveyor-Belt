import cv2
import os
import time

# Tạo thư mục để lưu frame nếu chưa có
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Mở camera
cap = cv2.VideoCapture("rtsp://Khangbede066:Khangbede066@192.168.1.55:554/stream1")  # 0 là camera mặc định
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

frame_count = 0
last_saved_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận frame từ camera")
        break
    
    current_time = time.time()
    if current_time - last_saved_time >= 1:
        # Lưu frame mỗi 1 giây
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
        last_saved_time = current_time
    
    # Hiển thị frame hiện tại
    cv2.imshow("Camera", frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
