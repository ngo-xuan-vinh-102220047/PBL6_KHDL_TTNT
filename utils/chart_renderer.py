import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from utils.config import get_config

def display_analysis_charts():
    """Hiển thị biểu đồ phân tích theo thời gian"""
    data = st.session_state.analysis_data
    
    if not data['timestamps']:
        st.warning("Không có dữ liệu phân tích để hiển thị")
        return
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Thời gian (s)': data['timestamps'],
        'Xác suất bạo lực': data['violence_probs'],
        'Điểm chuyển động': data['motion_scores'],
        'Trạng thái': data['detection_status']
    })
    
    # Main analysis chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Xác suất bạo lực theo thời gian', 'Điểm chuyển động theo thời gian'),
        vertical_spacing=0.1
    )
    
    # Add violence probability trace
    fig.add_trace(
        go.Scatter(
            x=df['Thời gian (s)'],
            y=df['Xác suất bạo lực'],
            mode='lines',
            name='Xác suất bạo lực',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ),
        row=1, col=1
    )
    
    # Add confidence threshold line
    confidence_threshold = get_config('CONFIDENCE_THRESHOLD')
    fig.add_hline(
        y=confidence_threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Ngưỡng tin cậy ({confidence_threshold})",
        row=1, col=1
    )
    
    # Add motion scores
    fig.add_trace(
        go.Scatter(
            x=df['Thời gian (s)'],
            y=df['Điểm chuyển động'],
            mode='lines',
            name='Điểm chuyển động',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,0,255,0.1)'
        ),
        row=2, col=1
    )
    
    # Add motion threshold line
    motion_threshold = get_config('MOTION_THRESHOLD')
    fig.add_hline(
        y=motion_threshold,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Ngưỡng chuyển động ({motion_threshold})",
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Phân tích bạo lực theo thời gian",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Thời gian (giây)", row=2, col=1)
    fig.update_yaxes(title_text="Xác suất", row=1, col=1)
    fig.update_yaxes(title_text="Điểm chuyển động", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Status distribution chart
    display_status_distribution(data)
    
    # Moving average chart
    display_moving_average(df)

def display_status_distribution(data):
    """Hiển thị biểu đồ phân bố trạng thái"""
    st.subheader("Phân bố trạng thái phát hiện")
    
    status_counts = pd.Series(data['detection_status']).value_counts()

    color_map = {
        'VIOLENCE': 'red',
        'FALSE ALARM': 'orange',
        'Normal': 'green'
    }
    colors = [color_map.get(label, 'gray') for label in status_counts.index]

    fig_pie = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=.3,
        marker=dict(colors=colors)
    )])
    
    fig_pie.update_layout(title="Tỉ lệ các trạng thái phát hiện")
    st.plotly_chart(fig_pie, use_container_width=True)

def display_moving_average(df):
    """Hiển thị biểu đồ trung bình động"""
    st.subheader("Xu hướng phát hiện (Trung bình động)")
    
    if len(df) > 10:
        window_size = min(get_config('CHART_WINDOW_SIZE'), len(df) // 4)
        df['Violence_MA'] = df['Xác suất bạo lực'].rolling(window=window_size).mean()
        df['Motion_MA'] = df['Điểm chuyển động'].rolling(window=window_size).mean()
        
        fig_ma = go.Figure()
        
        fig_ma.add_trace(go.Scatter(
            x=df['Thời gian (s)'],
            y=df['Violence_MA'],
            mode='lines',
            name=f'Xác suất bạo lực (MA{window_size})',
            line=dict(color='red', width=3)
        ))
        
        fig_ma.add_trace(go.Scatter(
            x=df['Thời gian (s)'],
            y=df['Motion_MA'],
            mode='lines',
            name=f'Chuyển động (MA{window_size})',
            line=dict(color='blue', width=3),
            yaxis='y2'
        ))
        
        fig_ma.update_layout(
            title=f"Xu hướng trung bình động (cửa sổ {window_size} frames)",
            xaxis_title="Thời gian (giây)",
            yaxis=dict(title="Xác suất bạo lực", side='left'),
            yaxis2=dict(title="Điểm chuyển động", side='right', overlaying='y'),
            showlegend=True
        )
        
        st.plotly_chart(fig_ma, use_container_width=True)

def display_detailed_report():
    """Hiển thị báo cáo chi tiết về phân tích"""
    data = st.session_state.analysis_data
    
    if not data['timestamps']:
        st.warning("Không có dữ liệu để tạo báo cáo")
        return
    
    df = pd.DataFrame(data)
    
    # Calculate statistics
    total_frames = len(df)
    violence_frames = len(df[df['detection_status'] == 'VIOLENCE'])
    false_alarm_frames = len(df[df['detection_status'] == 'FALSE ALARM'])
    normal_frames = len(df[df['detection_status'] == 'Normal'])
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tổng số frame", f"{total_frames:,}")
    with col2:
        st.metric("Frame bạo lực", f"{violence_frames} ({violence_frames/total_frames*100:.1f}%)")
    with col3:
        st.metric("Cảnh báo sai", f"{false_alarm_frames} ({false_alarm_frames/total_frames*100:.1f}%)")
    with col4:
        st.metric("Frame bình thường", f"{normal_frames} ({normal_frames/total_frames*100:.1f}%)")
    
    # Detailed statistics
    display_detailed_stats(df)
    
    # Timeline of events
    display_timeline_events(df)

def display_detailed_stats(df):
    """Hiển thị thống kê chi tiết"""
    st.subheader("Thống kê chi tiết")
    
    stats_col1, stats_col2 = st.columns(2)
    
    with stats_col1:
        st.write("**Xác suất bạo lực:**")
        st.write(f"- Trung bình: {df['violence_probs'].mean():.3f}")
        st.write(f"- Cao nhất: {df['violence_probs'].max():.3f}")
        st.write(f"- Thấp nhất: {df['violence_probs'].min():.3f}")
        st.write(f"- Độ lệch chuẩn: {df['violence_probs'].std():.3f}")
    
    with stats_col2:
        st.write("**Điểm chuyển động:**")
        st.write(f"- Trung bình: {df['motion_scores'].mean():.2f}")
        st.write(f"- Cao nhất: {df['motion_scores'].max():.2f}")
        st.write(f"- Thấp nhất: {df['motion_scores'].min():.2f}")
        st.write(f"- Độ lệch chuẩn: {df['motion_scores'].std():.2f}")

def display_timeline_events(df):
    """Hiển thị dòng thời gian sự kiện"""
    st.subheader("Dòng thời gian sự kiện")
    
    violence_frames = len(df[df['detection_status'] == 'VIOLENCE'])
    
    if violence_frames > 0:
        violence_periods = []
        current_start = None
        
        for i, (time, status) in enumerate(zip(df['timestamps'], df['detection_status'])):
            if status == 'VIOLENCE' and current_start is None:
                current_start = time
            elif status != 'VIOLENCE' and current_start is not None:
                violence_periods.append((current_start, df['timestamps'][i-1]))
                current_start = None
        
        # Handle case where violence continues to the end
        if current_start is not None:
            violence_periods.append((current_start, df['timestamps'].iloc[-1]))
        
        for i, (start, end) in enumerate(violence_periods, 1):
            duration = end - start
            st.write(f"**Sự kiện bạo lực #{i}:** {start:.1f}s - {end:.1f}s (Kéo dài: {duration:.1f}s)")
    else:
        st.success("🎉 Không phát hiện sự kiện bạo lực nào trong video!")