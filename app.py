import streamlit as st

# Import from local modules
from utils.config import initialize_session_state, initialize_model
from components.sidebar import render_sidebar
from components.upload_section import render_upload_section
from components.results_display import render_results

# ==========================================
# STREAMLIT APP CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Violence Detection System",
    page_icon="🚨",
    layout="wide"
)

# ==========================================
# MAIN APP
# ==========================================

def main():
    st.title("🚨 Hệ Thống Phát Hiện Bạo Lực Trong Video")
    st.markdown("""
    Phát hiện hành vi bạo lực sử dụng AI kết hợp phân tích chuyển động Optical Flow
    """)
    
    # Khởi tạo session state và model
    initialize_session_state()
    initialize_model()
    
    # Render các component
    render_sidebar()
    render_upload_section()
    render_results()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🎯 Hướng dẫn sử dụng:
    1. **Tải lên video** cần phân tích
    2. **Điều chỉnh tham số** trong sidebar nếu cần
    3. **Nhấn nút 'Bắt đầu phân tích'**
    4. **Xem kết quả** trong các tab và tải video đã xử lý

    ### 📊 Chú thích màu sắc:
    - 🔴 **Đỏ**: Phát hiện bạo lực (AI + Motion cao)
    - 🟠 **Cam**: Cảnh báo sai (AI detect nhưng motion thấp)
    - 🟢 **Xanh**: Bình thường
    """)

if __name__ == "__main__":
    main()