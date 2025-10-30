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
    page_title="Smart Traffic",
    page_icon="https://img.icons8.com/fluency/48/traffic-light.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# EARTHY NATURAL THEME CSS
# ============================================
st.markdown("""
<style>
    /* Icon library */
    @import url('https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.css');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root{
        --bg:#0f172a; /* slate-900 */
        --panel:rgba(255,255,255,0.06);
        --panel-border:rgba(255,255,255,0.08);
        --text:#e2e8f0; /* slate-200 */
        --muted:#94a3b8; /* slate-400 */
        --accent1:#6366f1; /* indigo-500 */
        --accent2:#22d3ee; /* cyan-400 */
        --success:#22c55e; /* green-500 */
        --warn:#f59e0b; /* amber-500 */
        --danger:#ef4444; /* red-500 */
    }

    .main {
        background: radial-gradient(1200px 600px at 10% -10%, rgba(34,211,238,0.08), transparent),
                    radial-gradient(1200px 600px at 110% 10%, rgba(99,102,241,0.10), transparent),
                    var(--bg);
        font-family: 'Inter', sans-serif;
    }

    .main *, .stMarkdown *, p, span, div, h1, h2, h3, h4, h5, h6, label {
        color: var(--text) !important;
    }

    .hero {
        border: 1px solid var(--panel-border);
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(34,211,238,0.12));
        padding: 2.2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35) inset, 0 10px 30px rgba(15,23,42,0.35);
        margin-bottom: 1.2rem;
    }
    .hero-title{
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        background: linear-gradient(135deg, var(--accent1), var(--accent2));
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub{ color: var(--muted) !important; margin-top:.4rem; }

    .glass-card {
        background: var(--panel);
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        border: 1px solid var(--panel-border);
        margin: 1rem 0;
        transition: transform .25s ease, box-shadow .25s ease;
        backdrop-filter: blur(10px);
    }
    .glass-card:hover { transform: translateY(-2px); }

    .metric-grid { display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:1rem; }
    @media (max-width: 1100px){ .metric-grid{ grid-template-columns: repeat(2, minmax(0,1fr)); } }
    @media (max-width: 700px){ .metric-grid{ grid-template-columns: 1fr; } }

    .metric-card { background: var(--panel); border:1px solid var(--panel-border); border-radius:14px; padding:1.4rem; text-align:center; }
    .metric-value { font-size:2.2rem; font-weight:800; background: linear-gradient(135deg, var(--accent1), var(--accent2)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .metric-label { font-size:.85rem; letter-spacing: .12em; color: var(--muted) !important; text-transform: uppercase; }

    .alert-high, .alert-medium, .alert-low {
        border-radius: 18px; padding: 2rem; text-align:center; border:1px solid var(--panel-border);
    }
    .alert-high { background: linear-gradient(135deg, rgba(239,68,68,0.10), rgba(99,102,241,0.08)); }
    .alert-medium { background: linear-gradient(135deg, rgba(245,158,11,0.10), rgba(99,102,241,0.08)); }
    .alert-low { background: linear-gradient(135deg, rgba(34,197,94,0.10), rgba(99,102,241,0.08)); }

    .recommendation-item { background: var(--panel); padding: 1rem 1.2rem; margin:.5rem 0; border-radius: 10px; border:1px solid var(--panel-border); color: var(--text) !important; }
    .recommendation-item:hover { transform: translateX(6px); }

    .stButton>button {
        background: linear-gradient(135deg, var(--accent1), var(--accent2));
        color: white !important; border: none; border-radius: 10px; padding: .85rem 1.2rem;
        font-size: 1rem; font-weight: 700; cursor: pointer; transition: transform .2s ease, box-shadow .2s ease; width: 100%;
        box-shadow: 0 10px 25px rgba(34,211,238,0.20);
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 14px 32px rgba(34,211,238,0.28); }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(2,6,23,0.95), rgba(2,6,23,0.85));
        border-right: 1px solid var(--panel-border);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: var(--text) !important; }

    .stSlider > div > div > div { background: linear-gradient(90deg, var(--accent1), var(--accent2)); }
    .stSlider label, .stSelectbox label { color: var(--muted) !important; font-weight: 600; }

    .stSuccess { background: rgba(34,197,94,0.12) !important; color: var(--text) !important; border: 1px solid rgba(34,197,94,0.35) !important; border-radius: 10px; }

    .footer { text-align: center; padding: 2rem 0; color: var(--muted) !important; border-top: 1px solid var(--panel-border); margin-top: 3rem; }
    .footer h3 { background: linear-gradient(135deg, var(--accent1), var(--accent2)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800; }

    .icon { display:inline-flex; align-items:center; gap:.5rem; }
    .icon i { font-size:1.1rem; }
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
st.markdown('<div class="hero"><div class="hero-title"><span class="icon"><i class="bi bi-traffic-light"></i></span> Smart City Traffic Manager</div><div class="hero-sub">AI-powered real-time traffic prediction and intelligent management dashboard</div></div>', unsafe_allow_html=True)

if not models_loaded:
    st.error("Models not found!")
    st.stop()

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("## <span class='icon'><i class='bi bi-sliders'></i></span> Traffic Parameters", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### <span class='icon'><i class='bi bi-cloud-sun'></i></span> Weather Conditions", unsafe_allow_html=True)
    temp_celsius = st.slider("Temperature (°C)", -30, 50, 15, 1)
    temp_kelvin = temp_celsius + 273.15
    rain = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0, 0.5)
    snow = st.slider("Snowfall (mm)", 0.0, 50.0, 0.0, 0.5)
    clouds = st.slider("Cloud Cover (%)", 0, 100, 50, 5)
    
    st.markdown("### <span class='icon'><i class='bi bi-clock-history'></i></span> Time Parameters", unsafe_allow_html=True)
    hour = st.slider("Hour", 0, 23, 8, 1)
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_name = st.selectbox("Day", day_names, index=1)
    day_of_week = day_names.index(day_name)
    is_holiday = st.radio("Holiday?", ['No', 'Yes'], index=0, horizontal=True)
    is_holiday_int = 1 if is_holiday == 'Yes' else 0
    
    st.markdown("---")
    predict_button = st.button("Predict Traffic", type="primary")
    
    if config:
        st.markdown("---")
        st.caption(f"**K:** {config['metrics']['optimal_k']} | **Acc:** {config['metrics']['test_accuracy']:.1%}")

# ============================================
# DASHBOARD - ONLY IF PREDICTION MADE
# ============================================
if predict_button:
    with st.spinner("Analyzing..."):
        result = predict_traffic(temp=temp_kelvin, rain=rain, snow=snow, clouds=clouds,
                                hour=hour, day_of_week=day_of_week, is_holiday=is_holiday_int)

    traffic_level = result['traffic_level']
    alert_class = "alert-high" if traffic_level == 'High' else "alert-medium" if traffic_level == 'Medium' else "alert-low"
    icon_class = "bi-exclamation-octagon-fill" if traffic_level == 'High' else ("bi-exclamation-triangle-fill" if traffic_level == 'Medium' else "bi-check-circle-fill")
    icon_color = "#ef4444" if traffic_level == 'High' else ("#f59e0b" if traffic_level == 'Medium' else "#22c55e")

    st.markdown(f"""
    <div class="{alert_class}">
        <h1 style="margin:0; font-size:3.6rem;"><i class="bi {icon_class}" style="color:{icon_color}"></i></h1>
        <h2 style="margin:0.5rem 0;">Traffic: {traffic_level}</h2>
        <p style="font-size:1.1rem; color:#cbd5e1">Confidence: {result['confidence']:.1%} &nbsp;•&nbsp; Congestion: {result['congestion_index']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # KPI cards
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

    # Tabs for details
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Probabilities", "Recommendations", "3D View"])

    with tab1:
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### <span class='icon'><i class='bi bi-card-checklist'></i></span> Current Conditions", unsafe_allow_html=True)
            st.markdown(f"<span class='icon'><i class='bi bi-thermometer-half'></i></span> <strong>{temp_celsius}°C</strong> | <span class='icon'><i class='bi bi-cloud-rain'></i></span> <strong>{rain}mm</strong> | <span class='icon'><i class='bi bi-snow'></i></span> <strong>{snow}mm</strong> | <span class='icon'><i class='bi bi-clouds'></i></span> <strong>{clouds}%</strong>", unsafe_allow_html=True)
            st.markdown(f"<span class='icon'><i class='bi bi-clock'></i></span> <strong>{hour:02d}:00 ({day_name})</strong> | <span class='icon'><i class='bi bi-calendar-event'></i></span> <strong>{is_holiday}</strong>", unsafe_allow_html=True)
            weather = "Clear" if rain == 0 and snow == 0 else "Adverse"
            st.success(f"Weather: {weather}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### <span class='icon'><i class='bi bi-cpu'></i></span> System Status", unsafe_allow_html=True)
            st.success("Active")
            st.metric("Models", "4/4")
            st.metric("Time", datetime.now().strftime("%H:%M:%S"))
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        probs = result['probabilities']
        colors = ['#22c55e' if k=='Low' else '#f59e0b' if k=='Medium' else '#ef4444' for k in probs.keys()]
        fig = go.Figure(data=[go.Bar(
            x=list(probs.keys()), y=list(probs.values()),
            marker=dict(color=colors),
            text=[f"{v:.1%}" for v in probs.values()], textposition='auto'
        )])
        fig.update_layout(height=380, yaxis=dict(range=[0, 1], tickformat='.0%'), template="plotly_dark",
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.02)')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### <span class='icon'><i class='bi bi-lightbulb'></i></span> Recommendations", unsafe_allow_html=True)
        icon_map = {
            'CRITICAL': 'bi-exclamation-octagon',
            'High': 'bi-exclamation-octagon',
            'Activate': 'bi-traffic-light',
            'alerts': 'bi-broadcast',
            'alt routes': 'bi-sign-railroad',
            'Peak': 'bi-clock',
            'Reduce speed': 'bi-speedometer2',
            'MODERATE': 'bi-exclamation-triangle',
            'Monitor': 'bi-eye',
            'Teams': 'bi-life-preserver',
            'OPTIMAL': 'bi-check-circle',
            'Maintenance': 'bi-tools',
            'freight': 'bi-truck'
        }
        for rec in result['recommendations']:
            clean = rec.encode('ascii', 'ignore').decode('ascii').strip()
            chosen = 'bi-lightbulb'
            for key, val in icon_map.items():
                if key.lower() in clean.lower():
                    chosen = val
                    break
            st.markdown(f'<div class="recommendation-item"><span class="icon"><i class="bi {chosen}"></i></span> {clean}</div>', unsafe_allow_html=True)

    with tab4:
        hours = np.linspace(0, 23, 24)
        clouds_grid = np.linspace(0, 100, 21)
        H, C = np.meshgrid(hours, clouds_grid)
        surface = (
            0.3
            + 0.35 * ((H >= 7) & (H <= 9))
            + 0.35 * ((H >= 17) & (H <= 19))
            + 0.05 * (C > 75)
        )
        surface = np.clip(surface, 0, 1)
        fig3d = go.Figure(data=[go.Surface(x=H, y=C, z=surface, colorscale=[[0, '#22c55e'], [0.5, '#f59e0b'], [1, '#ef4444']], showscale=False)])
        fig3d.update_layout(
            scene=dict(
                xaxis_title='Hour',
                yaxis_title='Cloud Cover (%)',
                zaxis_title='Congestion (norm)'
            ),
            height=420,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig3d, use_container_width=True)

else:
    # Show instruction before prediction
    st.info("Set traffic parameters in the sidebar and click 'Predict Traffic' to see results")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>Smart City Traffic Management</h3>
    <p><strong>ML-Powered</strong> | KNN + K-Means + Apriori</p>
    <p>© 2025 | Shivprasad & Parikshit | PCCOE</p>
</div>
""", unsafe_allow_html=True)
