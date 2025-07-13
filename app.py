import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Swiggy Sales App", layout="wide")

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
page = st.sidebar.radio("📍 Choose a page", [
    "🧮 Predict Sales",
    "📊 EDA Dashboard",
    "📘 Project Overview",
    "🧪 ML Pipeline"
])

# ------------------- 🧮 PREDICTION -------------------
if page == "🧮 Predict Sales":
    st.title("🛒 Swiggy Instamart Sales Predictor")
    with st.form("prediction_form"):
        st.subheader("📝 Enter Product & Outlet Info (in training order)")

        item_weight = st.number_input("1️⃣ Item Weight", min_value=0.0, step=0.1)
        item_fat_content = st.selectbox("2️⃣ Item Fat Content", ['Low Fat', 'Regular'])
        item_visibility = st.number_input("3️⃣ Item Visibility", min_value=0.0, step=0.01)
        item_type = st.selectbox("4️⃣ Item Type", [
            'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
            'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
            'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
            'Starchy Foods', 'Others', 'Seafood'])
        item_mrp = st.number_input("5️⃣ Item MRP", min_value=0.0, step=1.0)
        outlet_id = st.selectbox("6️⃣ Outlet Identifier", [
            'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
            'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'])
        outlet_size = st.selectbox("7️⃣ Outlet Size", ['Small', 'Medium', 'High'])
        outlet_loc = st.selectbox("8️⃣ Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.selectbox("9️⃣ Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])
        outlet_years = st.number_input("🔟 Outlet Years", min_value=0, step=1)
        item_category = st.selectbox("1️⃣1️⃣ Item Category Code (0: DR, 1: FD, 2: NC)", [0, 1, 2])

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

        # Download CSV
        output_df = df.copy()
        output_df['Predicted_Sales'] = [round(pred, 2)]
        st.download_button("⬇ Download Prediction", data=output_df.to_csv(index=False).encode(),
                           file_name="predicted_sales.csv", mime="text/csv")

# ------------------- 📊 EDA -------------------
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    uploaded_file = st.file_uploader("📁 Upload a CSV file for EDA", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("📌 Data Summary")
        st.write(df.describe())

        st.subheader("🎯 Column Distribution")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            col = st.selectbox("📈 Select numeric column", num_cols)
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig)

        st.subheader("📉 Correlation Heatmap")
        if len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.subheader("📌 Scatter Matrix")
        if len(num_cols) >= 2:
            fig = px.scatter_matrix(df[num_cols])
            st.plotly_chart(fig)

        st.subheader("🧮 Value Counts (Categorical Columns)")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            cat = st.selectbox("Select categorical column", cat_cols)
            fig = px.bar(df[cat].value_counts().reset_index(), x='index', y=cat,
                         labels={'index': cat, cat: 'Count'}, title=f"Value Counts of {cat}")
            st.plotly_chart(fig)

# ------------------- 📘 PROJECT OVERVIEW -------------------
elif page == "📘 Project Overview":
    st.title("📘 Project Overview")
    st.markdown("""
    ### 🎯 Purpose:
    This app helps predict Swiggy Instamart product sales based on item and outlet attributes using a trained machine learning model.

    ### 📦 Why We Built It:
    - Automate sales forecasting for inventory planning.
    - Help business managers make data-driven decisions.
    - Understand how product/outlet features affect sales.

    ### 📊 Key Objectives:
    - Build an end-to-end ML model
    - Provide real-time predictions
    - Include exploratory data analysis (EDA)
    - Deliver a user-friendly interface with download functionality

    ### 🧠 Technologies:
    - Python, Pandas, NumPy
    - XGBoost (Regression)
    - Streamlit (Frontend)
    - Plotly & Seaborn (Visualization)
    """)

# ------------------- 🧪 ML PIPELINE -------------------
elif page == "🧪 ML Pipeline":
    st.title("🧪 Machine Learning Pipeline")
    st.markdown("""
    ### 1️⃣ Data Collection:
    - Gathered historical product-level sales data from Swiggy Instamart.

    ### 2️⃣ Data Preprocessing:
    - Handled missing values (e.g., weight)
    - Label-encoded categorical features
    - Created derived features like `Outlet_Years`, `Item_Category`

    ### 3️⃣ Exploratory Data Analysis (EDA):
    - Used seaborn and matplotlib to visualize distributions, correlations, and categorical breakdowns.

    ### 4️⃣ Model Training:
    - Chose XGBoost Regressor based on high accuracy
    - Transformed target (`Sales`) using `log1p` for normalization

    ### 5️⃣ Model Evaluation:
    - Metrics: RMSE, R² Score
    - Final model saved using `joblib`

    ### 6️⃣ App Deployment:
    - Developed user-friendly Streamlit app
    - UI allows prediction + EDA + download
    """)

