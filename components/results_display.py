import streamlit as st
import os
from utils.config import get_config
from utils.video_processor import process_single_video
from utils.chart_renderer import display_analysis_charts, display_detailed_report

def render_results():
    """Render phần hiển thị kết quả"""
    # Process video button
    if hasattr(st.session_state, 'uploaded_file') and st.session_state.uploaded_file is not None:
        if st.button("🚀 Bắt đầu phân tích", type="primary"):
            process_video()
    
    # Display results if available
    if hasattr(st.session_state, 'processing_complete') and st.session_state.processing_complete:
        display_final_results()

def process_video():
    """Xử lý video"""
    model = st.session_state.model
    device = st.session_state.device
    
    if model is not None and device is not None:
        # Create output directory
        os.makedirs("outputs", exist_ok=True)
        
        # Generate output path
        uploaded_file = st.session_state.uploaded_file
        output_filename = f"processed_{uploaded_file.name}"
        output_path = os.path.join("outputs", output_filename)
        
        # Reset analysis data
        st.session_state.analysis_data = {
            'timestamps': [],
            'violence_probs': [],
            'motion_scores': [],
            'detection_status': [],
            'frame_times': []
        }
        
        # Process video
        with st.spinner("🔄 Đang phân tích video..."):
            try:
                process_single_video(
                    model=model,
                    device=device,
                    video_path=st.session_state.temp_video_path,
                    output_path=output_path,
                    confidence_threshold=get_config('CONFIDENCE_THRESHOLD'),
                    sequence_length=get_config('SEQUENCE_LENGTH'),
                    image_size=get_config('IMAGE_SIZE'),
                    motion_threshold=get_config('MOTION_THRESHOLD'),
                    analysis_data=st.session_state.analysis_data
                )
                
                st.session_state.processing_complete = True
                st.session_state.output_path = output_path
                st.session_state.output_filename = output_filename
                
            except Exception as e:
                st.error(f"❌ Lỗi trong quá trình xử lý: {e}")
        
        # Cleanup temp file
        if hasattr(st.session_state, 'temp_video_path'):
            os.unlink(st.session_state.temp_video_path)

def display_final_results():
    """Hiển thị kết quả cuối cùng"""
    st.success("✅ Phân tích hoàn tất!")
    
    # Display results in tabs
    tab1, tab2, tab3 = st.tabs(["🎥 Video kết quả", "📈 Biểu đồ phân tích", "📋 Báo cáo chi tiết"])
    
    with tab1:
        display_video_result()
    
    with tab2:
        display_analysis_charts()
    
    with tab3:
        display_detailed_report()

def display_video_result():
    """Hiển thị video kết quả"""
    st.subheader("📊 Video đã xử lý")
    st.video(st.session_state.output_path)
    
    # Download button
    with open(st.session_state.output_path, "rb") as file:
        st.download_button(
            label="📥 Tải video kết quả",
            data=file,
            file_name=st.session_state.output_filename,
            mime="video/mp4"
        )