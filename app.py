import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib  # 🌟 引入 joblib 用于加载安全的模型固化文件
import warnings
warnings.filterwarnings("ignore")

# ============================
# 0. Page Configuration
# ============================
st.set_page_config(page_title="Meniere's Disease Prediction Model", layout="wide")

st.title(" Vertigo Control Prediction Following EMS Surgery")
st.markdown("""
<div style='font-size: 16px; color: #555;'>
A machine learning-based clinical tool integrating radiological and clinical features to predict the probability of complete vertigo control following endolymphatic sac surgery in Meniere's Disease.
</div>
<hr>
""", unsafe_allow_html=True)

# ============================
# 1. Background Loading (🌟 0数据泄露的安全加载方式)
# ============================
@st.cache_resource
def load_models():
    # 直接从本地加载打包好的黑盒模型，彻底切断与原始 Excel 的联系
    calibrated_xgb = joblib.load('calibrated_model.pkl')
    base_xgb = joblib.load('base_model.pkl')
    
    # SHAP Explainer
    explainer = shap.Explainer(base_xgb)
    return calibrated_xgb, explainer

try:
    calibrated_model, explainer = load_models()
except Exception as e:
    st.error("Error loading models. Ensure 'calibrated_model.pkl' and 'base_model.pkl' are in the same directory as this app.py.")
    st.stop()

# 锁定特征名称
feature_names = ['VA-visibility', 'ED intensity', 'atva', 'QPVAA', 'PTA', 'Vertigo attack']

# ============================
# 2. Sidebar: Patient Characteristics
# ============================

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.header("Patient Characteristics")

# 定义临床标签映射字典
feature_mappings = {
    "VA-visibility": {1: "Clearly", 0: "Barely"},
    "ED intensity": {0: "Low", 1: "High"},
    "Vertigo attack": {1: "≤ 10 times", 2: "11-50 times", 3: "> 50 times"},
    "PTA": {0: "< 60 dBHL", 1: "≥ 60 dBHL"},
    "atva": {0: "< 120°", 1: "120°-140°", 2: "> 140°", 3: "> 140°"},
    "QPVAA": {0: "Other position", 1: "Posteroinferior position"}
}

# 🌟 预设选项数值 (不再依赖原始数据表读取)
feature_options = {
    "VA-visibility": [0, 1],
    "ED intensity": [0, 1],
    "atva": [0, 1, 2, 3],
    "QPVAA": [0, 1],
    "PTA": [0, 1],
    "Vertigo attack": [1, 2, 3]
}

user_inputs = {}
for feature in feature_names:
    options = feature_options[feature]
    
    # 侧边栏：映射选项为文本
    if feature in feature_mappings:
        user_inputs[feature] = st.sidebar.selectbox(
            f"{feature}:", 
            options=options,
            format_func=lambda x: feature_mappings[feature].get(x, str(x))
        )
    else:
        user_inputs[feature] = st.sidebar.selectbox(f"{feature}:", options=options)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_button = st.sidebar.button("Run Prediction", use_container_width=True)

# ============================
# 3. Main Panel: Results & Interpretation
# ============================
if predict_button:
    input_df = pd.DataFrame([user_inputs])
    
    with st.spinner('Calculating probabilities and generating SHAP explanations...'):
        
        # 1. Calculate Probabilities
        prob_1 = calibrated_model.predict_proba(input_df)[0, 1]
        prob_0 = 1.0 - prob_1
        
        # 🌟 动态判断指示灯和主题颜色
        if prob_0 >= 0.5:
            theme_color = "#28a745" # 绿灯
        else:
            theme_color = "#ffc107" # 黄灯
        
        col1, col2 = st.columns([1, 1.2])
        
        # 2. Display Prediction Outcome
        with col1:
            st.subheader("Prediction Outcome")
            
            # 🌟 注入发光指示灯
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 25px; border-radius: 8px; border-left: 5px solid {theme_color};'>
                <h3 style='margin-top: 0; color: #333; display: flex; align-items: center;'>
                    Probability of Complete Control
                    <span style='display: inline-block; width: 16px; height: 16px; border-radius: 50%; background-color: {theme_color}; box-shadow: 0 0 10px {theme_color}; margin-left: 12px;'></span>
                </h3>
                <h1 style='color: {theme_color}; margin-bottom: 0;'>{prob_0 * 100:.1f}%</h1>
                <p style='color: #666; font-size: 14px; margin-top: 5px;'>Probability of Incomplete Control (Failure Risk): {prob_1 * 100:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prob_0 >= 0.5:
                st.success("**Clinical Note:** The model suggests a favorable postoperative outcome, indicating a high likelihood of achieving complete vertigo control.")
            else:
                st.warning("**Clinical Note:** The model indicates a higher risk of incomplete vertigo control. Alternative interventions or conservative expectations may be warranted.")

        # 3. Display SHAP Waterfall Plot
        with col2:
            st.subheader("Local Interpretability (SHAP)")
            
            shap_values = explainer(input_df)
            
            if len(shap_values.shape) == 3:
                shap_obj = shap_values[0, :, 1]
            else:
                shap_obj = shap_values[0]
            
            mapped_data = []
            for i, feature in enumerate(feature_names):
                val = shap_obj.data[i]
                if feature in feature_mappings:
                    mapped_text = feature_mappings[feature].get(val, val)
                    mapped_data.append(mapped_text)
                else:
                    mapped_data.append(val)
            
            shap_obj.display_data = np.array(mapped_data, dtype=object)
            
            fig, ax = plt.subplots(figsize=(12, 4))
            shap.plots.waterfall(shap_obj, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            
            st.markdown("""
            <div style='font-size: 15px; color: #555; background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 3px solid #ffeeba;'>
            • <b><span style='color: #ff0051;'>Red bars</span></b> push the prediction towards <b>Incomplete Control</b> (higher failure risk).<br>
            • <b><span style='color: #008bfb;'>Blue bars</span></b> push the prediction towards <b>Complete Control</b> (better efficacy).
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Please configure the patient profiles in the sidebar and click **'Run Prediction'** to view the clinical assessment.")