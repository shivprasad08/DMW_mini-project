"""
Smart City Traffic Management - FIXED (No Empty Blocks)
========================================================
"""

import streamlit as st
import pickle
import json
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="🚦 Smart Traffic",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# EARTHY NATURAL THEME CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .main {
        background: #F4F1DE;
        font-family: 'Inter', sans-serif;
    }
    
    .main *, .stMarkdown *, p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: #3D405B !important;
    }
    
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #E07A5F 0%, #C96A52 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.15rem;
        color: #3D405B !important;
        margin-bottom: 2.5rem;
        font-weight: 500;
    }
    
    .glass-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 2px 12px rgba(61, 64, 91, 0.08);
        border: 1px solid rgba(224, 122, 95, 0.1);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(224, 122, 95, 0.12);
    }
    
    .glass-card h3 {
        color: #E07A5F !important;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1.2rem;
    }
    
    .glass-card p, .glass-card strong {
        color: #3D405B !important;
        font-size: 1rem;
        line-height: 1.8;
    }
    
    .metric-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(61, 64, 91, 0.08);
        border: 1px solid rgba(224, 122, 95, 0.15);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 24px rgba(224, 122, 95, 0.15);
    }
    
    .metric-value {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0.5rem 0;
        color: #E07A5F !important;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #3D405B !important;
        font-weight: 600;
    }
    
    .alert-high {
        background: linear-gradient(135deg, #E07A5F 0%, #D96B51 100%);
        border: 2px solid #C96A52;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 6px 24px rgba(224, 122, 95, 0.3);
    }
    
    .alert-high h1, .alert-high h2, .alert-high p {
        color: #FFFFFF !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .alert-medium {
        background: linear-gradient(135deg, #F2CC8F 0%, #E8C07D 100%);
        border: 2px solid #DFB96E;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 6px 24px rgba(242, 204, 143, 0.3);
    }
    
    .alert-medium h1, .alert-medium h2, .alert-medium p {
        color: #3D405B !important;
    }
    
    .alert-low {
        background: linear-gradient(135deg, #81B29A 0%, #6FA287 100%);
        border: 2px solid #5E9278;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 6px 24px rgba(129, 178, 154, 0.3);
    }
    
    .alert-low h1, .alert-low h2, .alert-low p {
        color: #FFFFFF !important;
    }
    
    .recommendation-item {
        background: #FFFFFF;
        padding: 1.2rem 1.5rem;
        margin: 0.6rem 0;
        border-radius: 10px;
        border-left: 4px solid #E07A5F;
        box-shadow: 0 2px 8px rgba(61, 64, 91, 0.06);
        color: #3D405B !important;
        font-size: 0.95rem;
        line-height: 1.6;
        transition: all 0.3s ease;
    }
    
    .recommendation-item:hover {
        transform: translateX(8px);
        box-shadow: 0 4px 16px rgba(224, 122, 95, 0.15);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #E07A5F 0%, #D96B51 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.85rem 2rem;
        font-size: 1rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(224, 122, 95, 0.3);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(224, 122, 95, 0.4);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F4F1DE 0%, #EBE7D0 100%);
        border-right: 1px solid rgba(224, 122, 95, 0.15);
    }
    
    section[data-testid="stSidebar"] * {
        color: #3D405B !important;
    }
    
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #E07A5F !important;
        font-weight: 700;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(224, 122, 95, 0.2);
        border-radius: 8px;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #3D405B !important;
    }
    
    ul[role="listbox"] {
        background-color: #FFFFFF !important;
        border: 1px solid rgba(224, 122, 95, 0.2);
    }
    
    ul[role="listbox"] li {
        background-color: #FFFFFF !important;
        color: #3D405B !important;
    }
    
    ul[role="listbox"] li:hover {
        background-color: #FEFDFB !important;
        border-left: 3px solid #E07A5F;
    }
    
    .stRadio > div {
        background-color: #FFFFFF;
        border: 1px solid rgba(224, 122, 95, 0.15);
        padding: 0.7rem;
        border-radius: 8px;
    }
    
    .stRadio label {
        color: #3D405B !important;
        font-weight: 600;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #E07A5F 0%, #81B29A 100%);
    }
    
    .stSlider label, .stSelectbox label {
        color: #3D405B !important;
        font-weight: 600;
    }
    
    .stSuccess {
        background: rgba(129, 178, 154, 0.15) !important;
        color: #3D405B !important;
        border: 1px solid rgba(129, 178, 154, 0.4) !important;
        border-radius: 8px;
    }
    
    .stSuccess * {
        color: #3D405B !important;
    }
    
    .footer {
        text-align: center;
        padding: 2.5rem 0;
        color: #3D405B !important;
        border-top: 1px solid rgba(224, 122, 95, 0.15);
        margin-top: 3rem;
        background: #FFFFFF;
        border-radius: 16px 16px 0 0;
    }
    
    .footer h3 {
        color: #E07A5F !important;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    try:
        with open('traffic_predictor.pkl', 'rb') as f:
            knn_model = pickle.load(f)
        with open('kmeans_model.pkl', 'rb') as f:
            kmeans_model = pickle.load(f)
        with open('scaler (3).pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('model_config.json', 'r') as f:
            config = json.load(f)
        return knn_model, kmeans_model, scaler, config, True
    except:
        return None, None, None, None, False

knn_model, kmeans_model, scaler, config, models_loaded = load_models()

def predict_traffic(temp, rain, snow, clouds, hour, day_of_week, is_holiday):
    is_peak_hour = 1 if hour in [7, 8, 9, 17, 18, 19] else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    base_congestion = 0.3
    if is_peak_hour: base_congestion += 0.35
    if rain > 0: base_congestion += min(rain * 0.03, 0.20)
    if snow > 0: base_congestion += min(snow * 0.05, 0.25)
    if clouds > 75: base_congestion += 0.05
    if is_weekend: base_congestion -= 0.15
    if is_holiday: base_congestion -= 0.10
    temp_celsius = temp - 273.15
    if temp_celsius < -5 or temp_celsius > 35: base_congestion += 0.08
    if hour >= 22 or hour <= 5: base_congestion -= 0.20
    congestion_index = np.clip(base_congestion, 0, 1)
    
    features = np.array([temp, rain, snow, clouds, hour, day_of_week,
                        is_peak_hour, is_weekend, is_holiday, congestion_index]).reshape(1, -1)
    features_scaled = features.copy()
    features_scaled[:, [0, 1, 2, 3]] = scaler.transform(features[:, [0, 1, 2, 3]])
    
    prediction = knn_model.predict(features_scaled)[0]
    probabilities = knn_model.predict_proba(features_scaled)[0]
    confidence = float(np.max(probabilities))
    class_proba = {cls: float(prob) for cls, prob in zip(knn_model.classes_, probabilities)}
    
    cluster = kmeans_model.predict(features_scaled)[0]
    cluster_labels = {0: 'Free-Flowing', 1: 'Moderate', 2: 'Congested', 3: 'Gridlocked'}
    cluster_label = cluster_labels.get(cluster, f'Cluster {cluster}')
    
    recommendations = []
    if prediction == 'High':
        recommendations = ["🚨 CRITICAL: High congestion expected", "🚦 Activate dynamic signals",
                          "📱 Send real-time alerts", "🛣️ Recommend alt routes"]
        if is_peak_hour: recommendations.append("⏰ Peak protocols")
        if rain > 0: recommendations.append("☔ Reduce speed limits")
    elif prediction == 'Medium':
        recommendations = ["⚠️ MODERATE: Traffic building", "👁️ Monitor closely", "🚑 Teams standby"]
    else:
        recommendations = ["✅ OPTIMAL: Low traffic", "🔧 Maintenance time", "🚛 Good for freight"]
    
    return {
        'traffic_level': prediction, 'confidence': confidence, 'probabilities': class_proba,
        'cluster': cluster_label, 'recommendations': recommendations, 'congestion_index': congestion_index
    }

# ============================================
# HEADER
# ============================================
st.markdown('<div class="main-header">🚦 Smart City Traffic Manager</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Real-Time Traffic Prediction & Management Dashboard</div>', unsafe_allow_html=True)

if not models_loaded:
    st.error("⚠️ Models not found!")
    st.stop()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## 🎛️ Traffic Parameters")
    st.markdown("---")
    
    st.markdown("### 🌤️ Weather Conditions")
    temp_celsius = st.slider("🌡️ Temperature (°C)", -30, 50, 15, 1)
    temp_kelvin = temp_celsius + 273.15
    rain = st.slider("🌧️ Rainfall (mm)", 0.0, 50.0, 0.0, 0.5)
    snow = st.slider("❄️ Snowfall (mm)", 0.0, 50.0, 0.0, 0.5)
    clouds = st.slider("☁️ Cloud Cover (%)", 0, 100, 50, 5)
    
    st.markdown("### ⏰ Time Parameters")
    hour = st.slider("🕐 Hour", 0, 23, 8, 1)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name = st.selectbox("📅 Day", day_names, index=1)
    day_of_week = day_names.index(day_name)
    is_holiday = st.radio("🎉 Holiday?", ['No', 'Yes'], index=0, horizontal=True)
    is_holiday_int = 1 if is_holiday == 'Yes' else 0
    
    st.markdown("---")
    predict_button = st.button("🚀 Predict Traffic", type="primary")
    
    if config:
        st.markdown("---")
        st.caption(f"**K:** {config['metrics']['optimal_k']} | **Acc:** {config['metrics']['test_accuracy']:.1%}")

# ============================================
# DASHBOARD - ONLY IF PREDICTION MADE
# ============================================
if predict_button:
    with st.spinner("🔄 Analyzing..."):
        result = predict_traffic(temp=temp_kelvin, rain=rain, snow=snow, clouds=clouds,
                                hour=hour, day_of_week=day_of_week, is_holiday=is_holiday_int)
    
    # Current Conditions and System Status AFTER prediction
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Current Conditions")
        st.markdown(f"**🌡️** {temp_celsius}°C | **🌧️** {rain}mm | **❄️** {snow}mm | **☁️** {clouds}%")
        st.markdown(f"**🕐** {hour:02d}:00 ({day_name}) | **🎉** {is_holiday}")
        weather = "Clear ☀️" if rain == 0 and snow == 0 else "Adverse ⛈️"
        st.success(f"**Weather:** {weather}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 System Status")
        st.success("✅ **Active**")
        st.metric("Models", "4/4")
        st.metric("Time", datetime.now().strftime("%H:%M:%S"))
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    traffic_level = result['traffic_level']
    alert_class = "alert-high" if traffic_level == 'High' else "alert-medium" if traffic_level == 'Medium' else "alert-low"
    icon = "🔴" if traffic_level == 'High' else "🟡" if traffic_level == 'Medium' else "🟢"
    
    st.markdown(f"""
    <div class="{alert_class}">
        <h1 style="margin:0; font-size:4rem;">{icon}</h1>
        <h2 style="margin:0.5rem 0;">Traffic: {traffic_level}</h2>
        <p style="font-size:1.2rem;">Confidence: {result['confidence']:.1%}</p>
        <p>Congestion: {result['congestion_index']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{result["confidence"]:.0%}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Pattern</div><div class="metric-value" style="font-size:1.5rem;">{result["cluster"]}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">High Risk</div><div class="metric-value">{result["probabilities"].get("High", 0):.0%}</div></div>', unsafe_allow_html=True)
    with col4:
        is_peak = hour in [7, 8, 9, 17, 18, 19]
        st.markdown(f'<div class="metric-card"><div class="metric-label">Peak Hour</div><div class="metric-value" style="font-size:2rem;">{"Yes" if is_peak else "No"}</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📊 Probabilities")
        probs = result['probabilities']
        fig = go.Figure(data=[go.Bar(
            x=list(probs.keys()), y=list(probs.values()),
            marker=dict(color=['#81B29A' if k=='Low' else '#F2CC8F' if k=='Medium' else '#E07A5F' for k in probs.keys()]),
            text=[f"{v:.1%}" for v in probs.values()], textposition='auto'
        )])
        fig.update_layout(height=350, yaxis=dict(range=[0, 1], tickformat='.0%'), template="plotly_white",
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#F4F1DE')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💡 Recommendations")
        for rec in result['recommendations']:
            st.markdown(f'<div class="recommendation-item">{rec}</div>', unsafe_allow_html=True)

else:
    # Show instruction before prediction
    st.info("👈 **Set traffic parameters in the sidebar and click 'Predict Traffic' to see results**")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🚦 Smart City Traffic Management</h3>
    <p><strong>ML-Powered</strong> | KNN + K-Means + Apriori</p>
    <p>© 2025 | Shivprasad & Parikshit | PCCOE</p>
</div>
""", unsafe_allow_html=True)
