import cv2
import torch
import numpy as np
from collections import deque
import streamlit as st
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.motion_analysis import calculate_motion_score

def process_single_video(model, device, video_path, output_path, 
                        confidence_threshold=0.85, sequence_length=16, 
                        image_size=64, motion_threshold=2.0, analysis_data=None):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"❌ Lỗi mở video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize variables
    frames_queue = []
    prev_frame_raw = None
    motion_scores = deque(maxlen=sequence_length)
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    chart_placeholder = st.empty()
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        frame_count += 1
        current_time = frame_count / fps if fps > 0 else frame_count / 30
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Đang xử lý frame {frame_count}/{total_frames} - Thời gian: {current_time:.1f}s")

        # Motion Calculation
        current_motion = calculate_motion_score(prev_frame_raw, frame)
        motion_scores.append(current_motion)
        avg_motion = np.mean(motion_scores) if len(motion_scores) > 0 else 0.0
        prev_frame_raw = frame.copy()

        # AI Preprocess
        try:
            resized = cv2.resize(frame, (image_size, image_size))
            normalized = resized / 255.0
            transposed = np.transpose(normalized, (2, 0, 1))
            frames_queue.append(transposed)
        except Exception as e:
            continue

        if len(frames_queue) > sequence_length:
            frames_queue.pop(0)

        # Logic Kết hợp
        label_text = "Initializing..."
        box_color = (255, 255, 0)  # Vàng
        violence_prob = 0.0
        detection_status = "Initializing"

        if len(frames_queue) == sequence_length:
            inp = torch.tensor(np.array([frames_queue]), dtype=torch.float32).to(device)
            with torch.no_grad():
                out_ai = model(inp)
                probs = torch.softmax(out_ai, dim=1)
                violence_prob = probs[0][1].item()

            is_ai_detect_violence = violence_prob > confidence_threshold
            is_motion_high = avg_motion > motion_threshold

            if is_ai_detect_violence:
                if is_motion_high:
                    label_text = f"VIOLENCE! ({violence_prob:.0%} | M:{avg_motion:.1f})"
                    box_color = (0, 0, 255)  # Đỏ
                    cv2.rectangle(frame, (0, 0), (width, height), box_color, 10)
                    detection_status = "VIOLENCE"
                else:
                    label_text = f"FALSE ALARM (M:{avg_motion:.1f})"
                    box_color = (0, 165, 255)  # Cam
                    detection_status = "FALSE ALARM"
            else:
                label_text = f"Normal (Conf:{violence_prob:.0%})"
                box_color = (0, 255, 0)  # Xanh lá
                detection_status = "Normal"
        else:
            detection_status = "Processing"

        # Store analysis data for charts
        if analysis_data is not None and detection_status != "Processing":
            analysis_data['timestamps'].append(current_time)
            analysis_data['violence_probs'].append(violence_prob)
            analysis_data['motion_scores'].append(avg_motion)
            analysis_data['detection_status'].append(detection_status)
            analysis_data['frame_times'].append(time.time() - start_time)

        # Vẽ thông tin lên frame
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
        
        # Motion Bar
        bar_len = int(min(avg_motion, 10.0) * 30)
        cv2.rectangle(frame, (20, 70), (20 + bar_len, 80), (255, 255, 255), -1)
        cv2.putText(frame, f"Motion: {avg_motion:.1f}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Time stamp
        cv2.putText(frame, f"Time: {current_time:.1f}s", (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        
        # Update real-time chart every 50 frames to avoid performance issues
        if frame_count % 50 == 0 and analysis_data and len(analysis_data['timestamps']) > 10:
            update_real_time_chart(chart_placeholder, analysis_data, confidence_threshold, motion_threshold)

    cap.release()
    out.release()
    progress_bar.empty()
    status_text.empty()
    chart_placeholder.empty()
    
    st.success(f"✅ Đã xử lý xong! Video được lưu tại: {output_path}")

def update_real_time_chart(placeholder, analysis_data, confidence_threshold, motion_threshold):
    """Cập nhật biểu đồ real-time"""
    data = analysis_data
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Xác suất bạo lực - Real Time', 'Điểm chuyển động - Real Time'),
        vertical_spacing=0.1
    )
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=data['violence_probs'],
            mode='lines',
            name='Xác suất bạo lực',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )
    
    # Add confidence threshold
    fig.add_hline(
        y=confidence_threshold,
        line_dash="dash",
        line_color="orange",
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['timestamps'],
            y=data['motion_scores'],
            mode='lines',
            name='Điểm chuyển động',
            line=dict(color='blue', width=2)
        ),
        row=2, col=1
    )
    
    # Add motion threshold
    fig.add_hline(
        y=motion_threshold,
        line_dash="dash", 
        line_color="green",
        row=2, col=1
    )
    
    fig.update_layout(height=400, showlegend=True)
    fig.update_xaxes(title_text="Thời gian (s)", row=2, col=1)
    fig.update_yaxes(title_text="Xác suất", row=1, col=1)
    fig.update_yaxes(title_text="Điểm chuyển động", row=2, col=1)
    
    placeholder.plotly_chart(fig, use_container_width=True)