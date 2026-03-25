import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://github.com/427Supraja/churn_prediction/blob/dac229ccc41501f9989f6b7306bb63a4d9392c04/image.jpeg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ================= PAGE CONFIG =================
st.set_page_config(page_title="SMART CHURN PREDICTION", layout="wide")
st.markdown("""
<div style='text-align:center;padding:15px'>
<h3>Predict Customer Churn with Explainable AI</h3>
<p>Enter customer details from the sidebar and click Predict to analyze churn risk, revenue impact, and retention strategy.</p>
</div>
""", unsafe_allow_html=True)
if "show_results" not in st.session_state:
    st.session_state.show_results = False

center = st.columns([2,1,2])
with center[1]:
    if st.button("🚀 Predict Churn"):
        st.session_state.show_results = True


# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Soft gradient background */
.stApp {
    background: linear-gradient(135deg,#e3f2fd,#fce4ec,#e8f5e9);
}

/* Title */
h1 {
    text-align:center;
    color:#2c3e50;
}

/* Metric cards */
.metric {
    padding:28px;
    border-radius:20px;
    text-align:center;
    background:white;
    box-shadow:0 6px 15px rgba(0,0,0,0.12);
    transition:0.3s;
}

.metric:hover {
    transform: scale(1.05);
}

/* Numbers */
.churn {color:#ef5350;font-size:32px;font-weight:bold;}
.stay {color:#66bb6a;font-size:32px;font-weight:bold;}
.rev {color:#42a5f5;font-size:28px;font-weight:bold;}

/* Button */
.stButton>button {
    background:linear-gradient(90deg,#90caf9,#ce93d8);
    color:black;
    border-radius:25px;
    height:48px;
    font-weight:bold;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background:white;
}

/* Rounded Inputs */
input, select {
    border-radius:14px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)


st.markdown("<h1>🛒 SMART CHURN PREDICTION</h1>", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model = pickle.load(open("model.pkl", "rb"))

# ================= SIDEBAR INPUTS =================
st.sidebar.header("CUSTOMER INPUTS")

Tenure = st.sidebar.number_input("Tenure (months)",0,100,1)
CityTier = st.sidebar.selectbox("City Tier",[1,2,3])
WarehouseToHome = st.sidebar.number_input("Warehouse Distance",0,50,25)
HourSpendOnApp = st.sidebar.number_input("Hours on App",0,10,1)
NumberOfDeviceRegistered = st.sidebar.number_input("Devices",1,10,1)
SatisfactionScore = st.sidebar.slider("Satisfaction Score",1,5,2)
OrderCount = st.sidebar.number_input("Order Count",0,100,1)
CashbackAmount = st.sidebar.number_input("Cashback Amount",0.0,10000.0,100.0)
AvgMonthlySpend = st.sidebar.number_input("Average Monthly Spend ₹",0,50000,1000)

# ================= PREDICTION BUTTON =================
#if st.sidebar.button("Predict Churn"):
if st.session_state.show_results:
    # ---- Create DataFrame ----
    input_df = pd.DataFrame([{
        "Tenure": Tenure,
        "CityTier": CityTier,
        "WarehouseToHome": WarehouseToHome,
        "HourSpendOnApp": HourSpendOnApp,
        "NumberOfDeviceRegistered": NumberOfDeviceRegistered,
        "SatisfactionScore": SatisfactionScore,
        "OrderCount": OrderCount,
        "CashbackAmount": CashbackAmount
    }])

    # ---- Predict ----
    prob = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    churn_prob = prob[classes.index(1)] * 100#==calculate churn probability in percentage form==#
    stay_prob = prob[classes.index(0)] * 100

    estimated_loss = AvgMonthlySpend * (churn_prob / 100)

    # ---- SHAP Calculation ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_vec = shap_values[1][0]
    else:
        shap_vec = shap_values[0]

    shap_vec = np.array(shap_vec).flatten() 
    shap_vec = shap_vec[:input_df.shape[1]]

    shap_df = pd.DataFrame({
        "Feature": input_df.columns.tolist(),
        "Impact": np.abs(shap_vec)
    }).sort_values(by="Impact", ascending=False)

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["📊 RESULTS", "🔍 EXPLANATION", "🛠 RETENTION"])

    # ---- TAB 1 RESULTS ----
    with tab1:
        c1,c2,c3 = st.columns(3)

        with c1:
            st.markdown(f"<div class='metric'><h4>🔥 Churn</h4><p class='churn'>{churn_prob:.2f}%</p></div>",unsafe_allow_html=True)

        with c2:
            st.markdown(f"<div class='metric'><h4>✅ Stay</h4><p class='stay'>{stay_prob:.2f}%</p></div>",unsafe_allow_html=True)

        with c3:
            st.markdown(f"<div class='metric'><h4>💰 Revenue Loss</h4><p class='rev'>₹{estimated_loss:.0f}</p></div>",unsafe_allow_html=True)

    # ---- TAB 2 SHAP ----
    with tab2:
        st.subheader("Feature Impact (SHAP)")
        st.dataframe(shap_df)
        st.bar_chart(shap_df.set_index("Feature")["Impact"])

    # ---- TAB 3 RETENTION ----
    with tab3:
        st.subheader("Recommended Retention Strategies")

        if churn_prob >= 30:
            if Tenure < 6:
                st.write("🎁 Provide loyalty offers")
            if SatisfactionScore <= 2:
                st.write("📞 Customer support follow-up")
            if OrderCount < 5:
                st.write("🔔 Personalized discounts")
            if CashbackAmount < 500:
                st.write("💰 Increase cashback incentives")
        else:
            st.balloons()
            st.success("🎉 Customer is low risk. Maintain engagement.")
            #st.success("Customer is low risk. Maintain engagement.")

# ================= FOOTER =================
st.markdown("""
<hr>
<center>© 2026 Smart Churn Prediction | ML + SHAP + Streamlit</center>
""", unsafe_allow_html=True)


