import cv2
import os
from tqdm import tqdm

# ==============================================================================
# PHẦN CÀI ĐẶT (BẠN CẦN CHỈNH SỬA Ở ĐÂY)
# ==============================================================================

# --- ĐƯỜNG DẪN THƯ MỤC ---
# Vui lòng thay đổi 2 đường dẫn dưới đây cho phù hợp với máy của bạn.
# Lưu ý quan trọng về cách viết đường dẫn:
# - Trên Windows: dùng dấu gạch chéo kép "\\" hoặc dấu gạch chéo đơn "/".
#   Ví dụ: "D:\\DuLieu\\VideoRaw" HOẶC "D:/DuLieu/VideoRaw"
# - Trên macOS/Linux: dùng dấu gạch chéo đơn "/".
#   Ví dụ: "/home/user/data/VideoRaw"

INPUT_ROOT_DIRECTORY = "PBL6_KHDL_TTNT\\VideoRaw"  # <-- THAY ĐỔI ĐƯỜNG DẪN THƯ MỤC VIDEO GỐC TẠI ĐÂY
OUTPUT_ROOT_DIRECTORY = "Frames_Output" # <-- THAY ĐỔI ĐƯỜNG DẪN THƯ MỤC LƯU ẢNH TẠI ĐÂY


# --- CÁC TÙY CHỈNH KHÁC ---
FRAME_INTERVAL = 15      # Cứ 15 frame thì lưu 1 ảnh
RESIZE_DIM = None        # Giữ kích thước gốc, hoặc đặt (224, 224)
IMAGE_FORMAT = ".jpg"    # Định dạng ảnh .jpg hoặc .png

# ==============================================================================
# PHẦN XỬ LÝ CHÍNH (Không cần chỉnh sửa)
# ==============================================================================

def process_single_video(video_path, output_video_dir):
    """Xử lý một file video duy nhất."""
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Lỗi: Không thể mở video {os.path.basename(video_path)}")
        return

    frame_count = 0
    saved_frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            if RESIZE_DIM:
                frame = cv2.resize(frame, RESIZE_DIM, interpolation=cv2.INTER_AREA)
            
            image_name = f"frame_{saved_frame_count:05d}{IMAGE_FORMAT}"
            image_path = os.path.join(output_video_dir, image_name)
            
            cv2.imwrite(image_path, frame)
            saved_frame_count += 1
        
        frame_count += 1
    
    cap.release()

def scan_and_process_folders(root_input_dir, root_output_dir):
    """
    Quét qua tất cả các thư mục con trong thư mục gốc, tìm và xử lý video.
    """
    print(f"Bắt đầu quét thư mục gốc: {root_input_dir}")
    
    for dirpath, dirnames, filenames in os.walk(root_input_dir):
        video_files = [f for f in filenames if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            continue

        for video_file in tqdm(video_files, desc=f"Xử lý thư mục '{os.path.basename(dirpath)}'"):
            video_path = os.path.join(dirpath, video_file)
            
            relative_path = os.path.relpath(dirpath, root_input_dir)
            output_parent_dir = os.path.join(root_output_dir, relative_path)
            
            video_name = os.path.splitext(video_file)[0]
            output_video_dir = os.path.join(output_parent_dir, video_name)
            
            process_single_video(video_path, output_video_dir)

    print("\nXử lý toàn bộ hoàn tất!")
    print(f"Tất cả ảnh đã được lưu tại: {root_output_dir}")

if __name__ == "__main__":
    # Kiểm tra xem thư mục đầu vào có tồn tại không
    if not os.path.isdir(INPUT_ROOT_DIRECTORY):
        print(f"LỖI: Thư mục đầu vào không tồn tại!")
        print(f"Vui lòng kiểm tra lại đường dẫn đã khai báo: {INPUT_ROOT_DIRECTORY}")
    else:
        print(f"Thư mục video gốc: {INPUT_ROOT_DIRECTORY}")
        print(f"Thư mục lưu ảnh: {OUTPUT_ROOT_DIRECTORY}")
        scan_and_process_folders(INPUT_ROOT_DIRECTORY, OUTPUT_ROOT_DIRECTORY)