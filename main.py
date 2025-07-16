import cv2
from queue import Queue
import threading
import torch
import time
import numpy as np

from ultralytics import YOLO
import supervision as sv
import serial.tools.list_ports

tracked_object = set()
object_timestamps = {}  # Lưu thời điểm vật thể chạm vào đường đỏ
object_positions = {}

class VideoCaptureThread:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.q = Queue(maxsize=5)
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
    
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.full():
                self.q.put(frame)
    
    def read(self):
        return self.q.get() if not self.q.empty() else None
    
    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

def get_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        return port.device

def connect_arduino():
    arduino_port = get_arduino_port()
    if arduino_port is None:
        print("❌ Không tìm thấy Arduino! Kiểm tra kết nối USB.")
        return None
    try:
        arduino = serial.Serial(arduino_port, 9600, timeout=1)
        time.sleep(2)
        print(f"✅ Kết nối Arduino thành công trên {arduino_port}!")
        return arduino
    except Exception as e:
        print(f"❌ Lỗi kết nối Arduino ({arduino_port}): {e}")
        return None

def belt(frame, crop_x_start, crop_x_end, crop_y_start, crop_y_end, rb=False):
    #Belt
    belt_frame = frame[crop_y_start-20:crop_y_end+20, crop_x_start-30:crop_x_end+20]

    if rb == False:
        # Áp dụng bộ lọc tăng cường sắc nét
        kernel_sharpening = np.array([[-1, -1, -1],
                                    [-1,  9, -1],
                                    [-1, -1, -1]])
        belt_frame = cv2.filter2D(belt_frame, -1, kernel_sharpening)

    return belt_frame

def count(frame, crop_x_start, crop_x_end, crop_y_start, crop_y_end):
    count_frame = frame[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Lấy kích thước màn hình (có thể thay bằng kích thước mong muốn)
    screen_width = 1080  # Thay bằng độ phân giải màn hình thực tế
    screen_height = 720

    # Resize frame để hiển thị toàn màn hình
    count_frame_resized = cv2.resize(count_frame, (screen_width, screen_height), interpolation=cv2.INTER_LINEAR)

    return count_frame_resized  # Trả về frame đã resize nếu cần dùng tiếp

def count_box(frame, count_list, font, font_scale, thickness):
    #Kích thước chữ
    (text_w_t, text_h_t), _ = cv2.getTextSize(f"Triangle: {count_list['triangle']}", font, font_scale, thickness)
    (text_w_r, text_h_r), _ = cv2.getTextSize(f"Rectangle: {count_list['rectangle']}", font, font_scale, thickness)

    #Vị trí đặt chữ
    org_tri = (10, 60)
    org_rect = (frame.shape[1]-450, 60)

    #Vẽ nền đen
    cv2.rectangle(frame, (org_tri[0] - 5, org_tri[1] - text_h_t - 5),
                    (org_tri[0] + text_w_t + 5, org_tri[1] + 20), (0, 0, 0), -1)
    cv2.rectangle(frame, (org_rect[0] - 5, org_rect[1] - text_h_r - 5),
                    (org_rect[0] + text_w_r + 5, org_rect[1] + 20), (0, 0, 0), -1)

    cv2.putText(img=frame,
                text=f"Triangle: {count_list['triangle']}",
                org=org_tri,
                fontFace=font,
                fontScale=font_scale,
                color=(0, 255, 0), 
                thickness=thickness)
    cv2.putText(img=frame,
                text=f"Rectangle: {count_list['rectangle']}",
                org=org_rect,
                fontFace=font,
                fontScale=font_scale,
                color=(0, 255, 0),
                thickness=thickness)

def show_frame(count_frame, combined_frame):
    list_frame_resized = []
    h = max(frame.shape[0] for frame in combined_frame)
    w = max(frame.shape[1] for frame in combined_frame)

    for frame in combined_frame:
        if len(frame.shape) == 2:
            list_frame_resized.append(cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_GRAY2BGR))
        else:
            list_frame_resized.append(cv2.resize(frame, (w, h)))
    
    combined_frame = np.hstack(list_frame_resized)

    cv2.imshow("Count", count_frame)
    cv2.imshow("Total", combined_frame)

def mouseMove(event, x, y):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"({x}, {y})")

def main():
    global tracked_object, object_timestamps
    font, font_scale, thickness = cv2.FONT_HERSHEY_TRIPLEX, 2, 1

    count_list = {
        'rectangle': 0,
        'triangle': 0
    }

    cap = VideoCaptureThread("rtsp://username:password@ip_address:554/stream1")
    # cap = cv2.VideoCapture(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚡ Running on: {device.upper()}")

    model = YOLO(r"model\best new v2.pt").to(device)
    arduino = connect_arduino()

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = sv.ByteTrack(track_activation_threshold=0.3, minimum_matching_threshold=0.8, lost_track_buffer=90)

    while True:
        frame = cap.read()

        if frame is None:
            continue

        # Define Frame
        crop_x_start_detect, crop_x_end_detect = 530, 780
        crop_y_start_detect, crop_y_end_detect = 500, 850

        #Belt Frame
        belt_frame = belt(frame.copy(), crop_x_start_detect, crop_x_end_detect, crop_y_start_detect, crop_y_end_detect, rb=True)

        # Count
        crop_x_start_count, crop_x_end_count = crop_x_start_detect - 150, crop_x_end_detect + 150
        crop_y_start_count, crop_y_end_count = crop_y_start_detect - 100, crop_y_end_detect
        count_frame = count(frame, crop_x_start_count, crop_x_end_count, crop_y_start_count, crop_y_end_count)

        # cv2.namedWindow('Frame')
        # cv2.setMouseCallback('Frame', mouseMove)
        # cv2.imshow("Frame", frame)

        # cv2.namedWindow('Count')
        # cv2.setMouseCallback('Count', mouseMove)

        result = model(belt_frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"{tracker_id}: {class_name} {confidence:.2f}"
            for tracker_id, class_name, confidence in zip(
                list(detections.tracker_id) if detections.tracker_id is not None else [],
                list(detections['class_name']) if detections["class_name"] is not None else [],
                list(detections.confidence) if detections.confidence is not None else []
            )
        ]

        # Tính vị trí dòng đỏ (ở giữa belt)
        line_y = 60
        cv2.line(belt_frame, (0, line_y), (count_frame.shape[1], line_y), (0, 0, 255), 2)  # Đường màu đỏ

        for obj_id, class_id, confidence, (x, y, _, _) in zip(detections.tracker_id, detections.class_id, detections.confidence, detections.xyxy):
            if confidence >= 0.8: #0.7
                if y <= line_y and obj_id not in tracked_object:
                    print(f"🔹 Vật thể {obj_id}: {class_id} {confidence} chạm vào đường!")

                    if obj_id not in object_timestamps:
                        object_timestamps[obj_id] = time.time()  # Lưu thời gian bắt đầu dừng

                if obj_id in object_timestamps or obj_id in object_positions:
                    elapsed_time = time.time() - object_timestamps[obj_id]
                    if elapsed_time >= 5.5: # Thời gian từ lúc nhận đến lúc gửi tín hiệu đến Arduino là 6.5s
                        print("Đã gửi tín hiệu.")
                        if arduino is not None:
                            if class_id == 0:
                                arduino.write(b'1')
                                count_list['rectangle'] += 1
                            else:
                                arduino.write(b'2')
                                count_list['triangle'] += 1 
                            print(f"🛠️ Gửi tín hiệu đến Arduino cho vật thể {obj_id}")

                        tracked_object.add(obj_id)
                        del object_timestamps[obj_id]  # Xóa khỏi danh sách chờ

            annotated_image = bounding_box_annotator.annotate(scene=belt_frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        count_box(count_frame, count_list, font, font_scale, thickness)
        show_frame(count_frame, [belt_frame])

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()

if __name__ == '__main__':
    main()