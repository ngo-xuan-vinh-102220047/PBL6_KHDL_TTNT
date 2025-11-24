import streamlit as st
import cv2
import tempfile
import os

def render_upload_section():
    """Render phần upload video"""
    st.header("📤 Tải lên video để phân tích")
    
    uploaded_file = st.file_uploader(
        "Chọn video file (MP4, AVI, MOV, MKV)",
        type=['mp4', 'avi', 'mov', 'mkv']
    )
    
    if uploaded_file is not None:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.read())
            st.session_state.temp_video_path = temp_file.name
        
        # Display original video and stats
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎬 Video gốc")
            st.video(st.session_state.temp_video_path)
        
        with col2:
            st.subheader("📊 Thống kê video")
            display_video_stats(st.session_state.temp_video_path)
        
        # Store uploaded file info
        st.session_state.uploaded_file = uploaded_file

def display_video_stats(video_path):
    """Hiển thị thống kê video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    st.metric("Tổng số frame", f"{total_frames:,}")
    st.metric("FPS", f"{fps:.1f}")
    st.metric("Thời lượng", f"{duration:.1f}s")
    cap.release()