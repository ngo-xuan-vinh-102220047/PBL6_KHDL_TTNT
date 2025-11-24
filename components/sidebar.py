import streamlit as st
from utils.config import get_config, update_config
import torch
import cv2

def render_sidebar():
    """Render thanh sidebar với các cài đặt"""
    st.sidebar.title("⚙️ Cấu hình hệ thống")
    
    # Model settings
    st.sidebar.subheader("Model Settings")
    
    model_weights = st.sidebar.text_input(
        "Đường dẫn weights model",
        value=get_config('MODEL_WEIGHTS_PATH')
    )
    update_config('MODEL_WEIGHTS_PATH', model_weights)
    
    confidence_threshold = st.sidebar.slider(
        "Ngưỡng tin cậy AI",
        min_value=0.1, max_value=1.0, 
        value=get_config('CONFIDENCE_THRESHOLD'), 
        step=0.05
    )
    update_config('CONFIDENCE_THRESHOLD', confidence_threshold)
    
    # Sequence settings
    sequence_length = st.sidebar.slider(
        "Độ dài chuỗi frame",
        min_value=8, max_value=32, 
        value=get_config('SEQUENCE_LENGTH'), 
        step=4
    )
    update_config('SEQUENCE_LENGTH', sequence_length)
    
    image_size = st.sidebar.slider(
        "Kích thước ảnh đầu vào",
        min_value=32, max_value=128, 
        value=get_config('IMAGE_SIZE'), 
        step=16
    )
    update_config('IMAGE_SIZE', image_size)
    
    # Motion settings
    motion_threshold = st.sidebar.slider(
        "Ngưỡng chuyển động",
        min_value=0.5, max_value=5.0, 
        value=get_config('MOTION_THRESHOLD'), 
        step=0.5
    )
    update_config('MOTION_THRESHOLD', motion_threshold)
    
    # Chart settings
    st.sidebar.subheader("Cài đặt biểu đồ")
    
    chart_window_size = st.sidebar.slider(
        "Kích thước cửa sổ biểu đồ",
        min_value=50, max_value=500, 
        value=get_config('CHART_WINDOW_SIZE'), 
        step=50
    )
    update_config('CHART_WINDOW_SIZE', chart_window_size)
    
    # System information
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ Thông tin hệ thống")
    
    if st.sidebar.button("Hiển thị thông tin hệ thống"):
        st.sidebar.text(f"PyTorch version: {torch.__version__}")
        st.sidebar.text(f"OpenCV version: {cv2.__version__}")
        st.sidebar.text(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.sidebar.text(f"GPU: {torch.cuda.get_device_name(0)}")