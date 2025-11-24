import cv2
import numpy as np

def calculate_motion_score(prev_frame, curr_frame):
    """
    Tính toán điểm chuyển động giữa hai frame sử dụng Optical Flow
    """
    if prev_frame is None or curr_frame is None:
        return 0.0
    
    h, w = prev_frame.shape[:2]
    scale = 320 / w 
    
    # Resize để tăng tốc tính toán
    prev_small = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)
    curr_small = cv2.resize(curr_frame, (0, 0), fx=scale, fy=scale)
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_small, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_small, cv2.COLOR_BGR2GRAY)
    
    # Calculate Optical Flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Calculate magnitude
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)