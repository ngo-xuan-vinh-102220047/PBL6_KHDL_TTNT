import streamlit as st
import torch
import torch.nn as nn
import os
from models.violence_detector import ViolenceDetector

# Default configuration
DEFAULT_CONFIG = {
    'MODEL_WEIGHTS_PATH': "weights/best_model_pytorch.pth",
    'CONFIDENCE_THRESHOLD': 0.85,
    'SEQUENCE_LENGTH': 16,
    'IMAGE_SIZE': 64,
    'MOTION_THRESHOLD': 2.0,
    'CHART_WINDOW_SIZE': 200
}

def initialize_session_state():
    """Khởi tạo session state"""
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {
            'timestamps': [],
            'violence_probs': [],
            'motion_scores': [],
            'detection_status': [],
            'frame_times': []
        }
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model = None
        st.session_state.device = None
    
    # Initialize config in session state
    for key, value in DEFAULT_CONFIG.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def initialize_model():
    """Khởi tạo model với cache"""
    if not st.session_state.model_loaded:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"🔄 Đang khởi tạo model trên thiết bị: {device}")
        
        model = ViolenceDetector()
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        model.to(device)
        
        # Load weights
        weights_path = st.session_state['MODEL_WEIGHTS_PATH']
        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=device)
                # Handle DataParallel prefix
                if list(state.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
                    new_state = {k.replace("module.", ""): v for k, v in state.items()}
                    model.load_state_dict(new_state)
                else:
                    model.load_state_dict(state)
                
                model.eval()
                st.session_state.model_loaded = True
                st.session_state.model = model
                st.session_state.device = device
                st.success("✅ Model đã được tải thành công!")
            except Exception as e:
                st.error(f"❌ Lỗi khi tải model: {e}")
        else:
            st.error(f"⚠️ Không tìm thấy file weights: {weights_path}")

def get_config(key):
    """Lấy giá trị cấu hình từ session state"""
    return st.session_state.get(key, DEFAULT_CONFIG.get(key))

def update_config(key, value):
    """Cập nhật giá trị cấu hình"""
    st.session_state[key] = value