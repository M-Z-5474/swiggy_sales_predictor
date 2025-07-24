import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Swiggy Sales Predictor", layout="wide")

# ------------------- LOAD MODEL & ENCODERS -------------------
model = joblib.load("xgboost_final_model.pkl")
encoder_files = [
    'le_Item_Fat_Content.pkl', 'le_Item_Type.pkl', 'le_Outlet_Identifier.pkl',
    'le_Outlet_Size.pkl', 'le_Outlet_Location_Type.pkl', 'le_Outlet_Type.pkl'
]
encoders = {file.replace("le_", "").replace(".pkl", ""): joblib.load(file) for file in encoder_files}

feature_order = [
    'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
    'Item_MRP', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
    'Outlet_Type', 'Outlet_Years', 'Item_Category'
]

# ------------------- SIDEBAR -------------------
page = st.sidebar.radio("📍 Choose a Page", [
    "🧮 Predict Sales",
    "📘 Project Overview",
    "ℹ️ About"
])

# ------------------- 🧮 PREDICT SALES -------------------
if page == "🧮 Predict Sales":
    st.title("🛒 Swiggy Instamart Sales Predictor")
    with st.form("prediction_form"):
        st.subheader("📝 Enter Product & Outlet Info")

        item_weight = st.number_input("Item Weight", min_value=0.0, step=0.1)
        item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
        item_visibility = st.number_input("Item Visibility", min_value=0.0, step=0.01)
        item_type = st.selectbox("Item Type", [
            'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
            'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
            'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
            'Starchy Foods', 'Others', 'Seafood'])
        item_mrp = st.number_input("Item MRP", min_value=0.0, step=1.0)
        outlet_id = st.selectbox("Outlet Identifier", [
            'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
            'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'])
        outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
        outlet_loc = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])
        outlet_years = st.number_input("Outlet Years", min_value=0, step=1)
        item_category = st.selectbox("Item Category Code (0: DR, 1: FD, 2: NC)", [0, 1, 2])

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        user_input = {
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_id,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_loc,
            'Outlet_Type': outlet_type,
            'Outlet_Years': outlet_years,
            'Item_Category': item_category
        }

        df = pd.DataFrame([user_input])
        for col in encoders:
            df[col] = encoders[col].transform(df[col])

        df = df[feature_order]
        pred_log = model.predict(df)
        pred = np.expm1(pred_log[0])

        st.success(f"💰 **Predicted Sales: ₹{round(pred, 2)}**")

        output_df = df.copy()
        output_df['Predicted_Sales'] = [round(pred, 2)]
        st.download_button("⬇ Download Prediction", data=output_df.to_csv(index=False).encode(),
                           file_name="predicted_sales.csv", mime="text/csv")

# ------------------- 📘 PROJECT OVERVIEW -------------------
elif page == "📘 Project Overview":
    st.title("📘 Project Overview")

    st.markdown("""
    ### 🎯 Purpose:
    This AI-powered app predicts Swiggy Instamart product sales based on item and outlet characteristics using a trained machine learning model.

    ### 📦 Use Case:
    - Forecast product-level sales for better inventory planning
    - Support data-driven decision-making for business managers
    - Showcase ML deployment using Streamlit

    ### 🧠 Machine Learning Pipeline:
    **1️⃣ Data Preprocessing:**
    - Imputed missing values (e.g., `Item_Weight`)
    - Encoded categorical variables
    - Engineered features like `Outlet_Years` and `Item_Category`

    **2️⃣ Model Training:**
    - Model Used: **XGBoost Regressor**
    - Target column transformed with log1p
    - Model saved using `joblib`

    **3️⃣ Evaluation:**
    - RMSE and R² used for accuracy
    - Model tested on unseen data

    **4️⃣ Deployment:**
    - Deployed using **Streamlit**
    - Supports real-time input + prediction
    """)

# ------------------- ℹ️ ABOUT -------------------
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")

    st.markdown("""
    **🛒 App Name:** Swiggy Instamart Sales Predictor  
    **📌 Description:** A machine learning web app that predicts product sales based on outlet and item features using a trained XGBoost model.  
  
### 👨‍💻 Developer Info

**Muhammad Zain Mushtaq**  
AI/ML & Data Science Enthusiast | IT Graduate  
📍 Pakistan  
🔗 [GitHub](https://github.com/M-Z-5474) | [LinkedIn](https://www.linkedin.com/in/muhammad-zain-m-a75163358/)
""")

# Footer
st.markdown("---")
st.markdown(
    "Built with ❤️ by [Muhammad Zain Mushtaq](https://github.com/M-Z-5474) • "
    "[View GitHub Repo](https://github.com/M-Z-5474/swiggy_sales_predictor)"
)

