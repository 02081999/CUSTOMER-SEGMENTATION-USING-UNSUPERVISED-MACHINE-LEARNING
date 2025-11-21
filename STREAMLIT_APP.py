import os
import joblib
import streamlit as st
import numpy as np

# ============================================================
# 🛍️ CUSTOMER SEGMENTATION — STREAMLIT APP (ENHANCED VERSION)
# ============================================================

st.set_page_config(
    page_title="🛍️ Customer Segmentation Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------
# 🔹 Load Model + Scaler
# ---------------------------------------------
@st.cache_resource
def load_model_and_scaler():
    base_dir = os.path.dirname(__file__)
    kmeans = joblib.load(os.path.join(base_dir, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    return kmeans, scaler

kmeans, scaler = load_model_and_scaler()

# ============================================================
# 🔍 CLUSTER DETAILS (Centroids + Business Strategies)
# ============================================================

cluster_centroids = {
    0: {
        "income": 55, "spend": 50,
        "name": "Balanced Customers",
        "desc": "Mid Income + Mid Spending — stable and average shoppers.",
        "strategy": [
            "Upsell through complementary products",
            "Provide loyalty points to increase retention",
            "Give seasonal or festival-based offers"
        ]
    },

    1: {
        "income": 87, "spend": 82,
        "name": "Premium Customers",
        "desc": "High Income + High Spending — loyal and highly profitable.",
        "strategy": [
            "Exclusive VIP membership programs",
            "Premium or luxury product upgrades",
            "Invite-only events or early access sales"
        ]
    },

    2: {
        "income": 26, "spend": 79,
        "name": "Impulsive Buyers",
        "desc": "Low Income + High Spending — trend-driven and offer-sensitive.",
        "strategy": [
            "Flash deals and limited-time offers",
            "Product bundles to increase cart size",
            "Influencer-based marketing and trending items"
        ]
    },

    3: {
        "income": 88, "spend": 17,
        "name": "Cautious High Earners",
        "desc": "High Income + Low Spending — spend carefully despite wealth.",
        "strategy": [
            "Value-focused communication (show benefits)",
            "Free demos or trials to build trust",
            "Follow-up reminders to reduce hesitation"
        ]
    },

    4: {
        "income": 26, "spend": 21,
        "name": "Budget Customers",
        "desc": "Low Income + Low Spending — highly price-sensitive shoppers.",
        "strategy": [
            "Regular discounts and price drops",
            "Essential collections and basic product ranges",
            "Low-cost subscription or EMI-based plans"
        ]
    }
}


# Why the customer belongs to the predicted segment
def explain_membership(cluster, income, score):
    cent = cluster_centroids[cluster]
    return (
        f"### 📌 Why this customer belongs to **{cent['name']}**\n"
        f"- Model compares customer's income and spending with cluster centroids.\n"
        f"- This cluster's average income ≈ **{cent['income']}k$**.\n"
        f"- This cluster's average spending score ≈ **{cent['spend']}**.\n\n"
        f"Your customer's values:\n"
        f"- Income: **{income}k$**\n"
        f"- Spending Score: **{score}**\n\n"
        f"Since these values closely match this cluster’s profile, the model assigns them here."
    )


# ============================================================
# 📘 Sidebar — Full Guidance for Non-Technical Users
# ============================================================

with st.sidebar:
    st.title("📘 Understanding Segments")

    st.info("""
    This model groups customers into **5 segments** using **K-Means Clustering**.

    ### 🔍 How customers are grouped?
    Customers with **similar Annual Income & Spending Score** fall into the same cluster.

    ### 📌 What ranges does dataset generally have?
    - **Income Range:** 15k$ – 137k$
    - **Spending Score Range:** 1 – 100
    - **Age Range:** 18 – 70

    ### 📌 Why only Income & Spending Score?
    These two features create the clearest natural groups in the customer dataset.
    """)

    st.markdown("---")
    st.caption("Developed for Customer Segmentation Project (K-Means)")

# ============================================================
# 🔢 Input Section
# ============================================================
st.title("🛍️ Customer Segmentation Predictor")
st.write("Enter customer details below to predict which segment they belong to.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender (not used in clustering)", ["0 = Male", "1 = Female"])
    income = st.number_input(
        "Annual Income (k$)",
        min_value=10.0,
        max_value=140.0,
        value=60.0,
        help="Dataset income generally ranges between 15k$ and 137k$"
    )

with col2:
    age = st.slider("Age", 18, 70, 30, help="Age is not used for clustering but shown for reference")
    score = st.slider(
        "Spending Score (1–100)",
        1, 100, 50,
        help="Higher score = customer spends more frequently"
    )

# ============================================================
# 🔮 Prediction Section
# ============================================================
if st.button("🔍 Predict Customer Segment"):
    features = np.array([[income, score]])
    scaled = scaler.transform(features)
    cluster = int(kmeans.predict(scaled)[0])

    cent = cluster_centroids[cluster]

    st.markdown(f"## 🧩 Predicted Segment: **{cent['name']}**")
    st.info(cent['desc'])

    # Why customer belongs here
    st.markdown(explain_membership(cluster, income, score))

    # -------------------------------
    # ⭐ Recommended Business Strategies
    # -------------------------------
    st.markdown("## 🎯 Recommended Business Strategies")
    for s in cent["strategy"]:
        st.markdown(f"- {s}")

    st.success("Segmentation completed successfully!")
    st.balloons()

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption("© 2025 Shopper Spectrum | Built with ❤️ using Machine Learning + Streamlit")
