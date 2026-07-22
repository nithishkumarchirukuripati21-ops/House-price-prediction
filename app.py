import streamlit as st
import pandas as pd
import joblib
import shap


st.set_page_config(
    page_title="ValuaHome | AI Property Estimator",
    page_icon="🏠",
    layout="wide"
)


st.markdown("""
<style>

    .stApp {
        background-color: #0F172A !important;
        color: #F8FAFC !important;
    }

    h1, h2, h3, h4, h5, h6, .hero-title {
        color: #FFFFFF !important;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8 !important;
        margin-bottom: 1.5rem;
    }


    label, p, span, div[data-testid="stWidgetLabel"] {
        color: #CBD5E1 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div, input, textarea {
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border-color: #334155 !important;
    }

    
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        color: #94A3B8 !important;
    }
    button[aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom-color: #38BDF8 !important;
    }


    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    .price-badge {
        background: linear-gradient(135deg, #0284C7 0%, #38BDF8 100%);
        color: #FFFFFF !important;
        padding: 1.25rem 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 20px -3px rgba(56, 189, 248, 0.3);
    }
    
    .price-badge * {
        color: #FFFFFF !important;
    }
    
    .price-title {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.95;
    }
    
    .price-value {
        font-size: 2.75rem;
        font-weight: 800;
        margin-top: 0.2rem;
    }

    /* Metric Card Custom Overrides */
    div[data-testid="stMetricValue"] {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_assets():
    model = joblib.load("models/best_xgboost_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    metadata = joblib.load("models/model_info.pkl")
    explainer = joblib.load("models/shap_explainer.pkl")
    return model, feature_columns, metadata, explainer

try:
    model, feature_columns, metadata, explainer = load_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False


with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/home.png", width=70)
    st.title("ValuaHome Engine")
    st.caption("v2.4 • XGBoost Regressor")
    
    st.divider()
    
    if model_loaded:
        st.success("🟢 Model Pipeline Online")
    else:
        st.error("🔴 Failed to load model artifacts")
        st.stop()
        
    st.subheader("📊 Engine Specs")
    st.text(f"• Dataset: Ames Housing\n• Features: {len(feature_columns)} variables\n• Explainer: TreeSHAP")
    
    st.divider()
    
    
    st.subheader("⚡ Quick Presets")
    if st.button("Standard Family Home", use_container_width=True):
        st.session_state["qual"] = 6
        st.session_state["liv_area"] = 1800
        st.session_state["ext_qual"] = "Good"
    if st.button("Luxury Estate", use_container_width=True):
        st.session_state["qual"] = 9
        st.session_state["liv_area"] = 3500
        st.session_state["ext_qual"] = "Excellent"


st.markdown('<div class="hero-title">🏠 Residential Valuation Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Fill in the property metrics below to calculate instant market value estimates.</div>', unsafe_allow_html=True)


input_tab1, input_tab2, input_tab3 = st.tabs([
    "📐 Dimensions & Layout", 
    "✨ Quality & Finishes", 
    "📍 Location & Utilities"
])

with input_tab1:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        overall_qual = st.slider(
            "Overall Quality (1-10)", 
            min_value=1, max_value=10, 
            value=st.session_state.get("qual", 5)
        )
        gr_liv_area = st.number_input(
            "Above Ground Living Area (sq ft)", 
            min_value=300, max_value=6000, 
            value=st.session_state.get("liv_area", 1500), step=50
        )
        first_flr_sf = st.number_input(
            "First Floor Area (sq ft)", 
            min_value=300, max_value=4000, value=1000, step=50
        )
    with col2:
        second_flr_sf = st.number_input(
            "Second Floor Area (sq ft)", 
            min_value=0, max_value=3000, value=500, step=50
        )
        full_bath = st.select_slider("Full Bathrooms", options=[0, 1, 2, 3, 4, 5], value=2)
        fireplaces = st.select_slider("Fireplaces", options=[0, 1, 2, 3, 4, 5], value=1)

with input_tab2:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        exter_qual = st.selectbox(
            "Exterior Quality", 
            ["Excellent", "Good", "Average", "Fair", "Poor"], 
            index=["Excellent", "Good", "Average", "Fair", "Poor"].index(st.session_state.get("ext_qual", "Average"))
        )
        kitchen_qual = st.selectbox("Kitchen Quality", ["Excellent", "Good", "Average", "Fair"], index=2)
        bsmt_exposure = st.selectbox("Basement Exposure", ["Good Exposure", "Average Exposure", "Minimum Exposure", "No Exposure"], index=3)
    with col2:
        garage_cars = st.select_slider("Garage Capacity (Cars)", options=[0, 1, 2, 3, 4, 5], value=2)
        garage_finish = st.selectbox("Garage Finish Status", ["Finished", "Rough Finished", "Unfinished"], index=1)
        garage_type = st.selectbox("Garage Structure", ["Attached", "Detached", "Built-In", "Car Port", "Basement", "Two Types"])

with input_tab3:
    col1, col2 = st.columns(2, gap="large")
    with col1:
        neighborhood = st.selectbox(
            "Neighborhood Location", 
            ["North Ames", "College Creek", "Old Town", "Edwards", "Somerset", "Northridge Heights"]
        )
    with col2:
        central_air = st.radio("Central Air Conditioning", ["Yes", "No"], horizontal=True)

st.divider()

col_btn, _ = st.columns([1, 2])
with col_btn:
    predict_btn = st.button("🔮 Calculate Market Valuation", type="primary", use_container_width=True)

if predict_btn:
    exter_mapping = {"Excellent": "Ex", "Good": "Gd", "Average": "TA", "Fair": "Fa", "Poor": "Po"}
    kitchen_mapping = {"Excellent": "Ex", "Good": "Gd", "Average": "TA", "Fair": "Fa"}
    central_air_mapping = {"Yes": "Y", "No": "N"}
    garage_finish_mapping = {"Finished": "Fin", "Rough Finished": "RFn", "Unfinished": "Unf"}
    garage_type_mapping = {"Attached": "Attchd", "Detached": "Detchd", "Built-In": "BuiltIn", "Car Port": "CarPort", "Basement": "Basment", "Two Types": "2Types"}
    bsmt_exposure_mapping = {"Good Exposure": "Gd", "Average Exposure": "Av", "Minimum Exposure": "Mn", "No Exposure": "No"}
    neighborhood_mapping = {"North Ames": "NAmes", "College Creek": "CollgCr", "Old Town": "OldTown", "Edwards": "Edwards", "Somerset": "Somerst", "Northridge Heights": "NridgHt"}

    input_data = pd.DataFrame({
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "1stFlrSF": [first_flr_sf],
        "2ndFlrSF": [second_flr_sf],
        "GarageCars": [garage_cars],
        "FullBath": [full_bath],
        "Fireplaces": [fireplaces],
        "ExterQual": [exter_mapping[exter_qual]],
        "KitchenQual": [kitchen_mapping[kitchen_qual]],
        "CentralAir": [central_air_mapping[central_air]],
        "GarageFinish": [garage_finish_mapping[garage_finish]],
        "GarageType": [garage_type_mapping[garage_type]],
        "BsmtExposure": [bsmt_exposure_mapping[bsmt_exposure]],
        "Neighborhood": [neighborhood_mapping[neighborhood]]
    })

    # Prepare features
    input_encoded = pd.get_dummies(input_data)
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    # Predict
    prediction = model.predict(input_encoded)[0]

    # Display Valuation Card
    res_col1, res_col2 = st.columns([1, 1.5], gap="large")
    
    with res_col1:
        st.markdown(f"""
        <div class="price-badge">
            <div class="price-title">Estimated Market Value</div>
            <div class="price-value">${prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col2:
        st.markdown("#### 💡 Property Summary Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Price / SqFt", f"${prediction / max(gr_liv_area, 1):,.2f}")
        m2.metric("Overall Rank", f"{overall_qual}/10")
        m3.metric("Bathrooms", f"{full_bath}")

    st.divider()

    st.subheader("🔍 Valuation Feature Breakdown (TreeSHAP)")
    
    shap_values = explainer.shap_values(input_encoded)
    shap_df = pd.DataFrame({
        "Feature": input_encoded.columns,
        "Impact ($)": shap_values[0]
    }).sort_values(by="Impact ($)", key=abs, ascending=False)

    exp_col1, exp_col2 = st.columns([1.5, 1], gap="large")

    with exp_col1:
        top_8 = shap_df.head(8).set_index("Feature")
        st.bar_chart(top_8["Impact ($)"], color="#38BDF8")

    with exp_col2:
        st.dataframe(
            shap_df.head(8).reset_index(drop=True),
            use_container_width=True,
            column_config={
                "Impact ($)": st.column_config.NumberColumn(
                    "Price Contribution", 
                    format="$%,.0f"
                )
            }
        )