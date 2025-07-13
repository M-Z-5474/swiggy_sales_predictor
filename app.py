import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import plotly.express as px
from io import BytesIO

# Set page config
st.set_page_config(page_title="Swiggy Sales App", layout="wide")

# Background image
def set_bg():
    bg_img = "https://free-barcode.com/barcode/inventory-management/key-components-regression-analysis-sales-forecasting/2.jpg"
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url('{bg_img}');
            background-size: cover;
            background-attachment: fixed;
            color: #ffffff;
        }}
        .stButton > button {{
            color: white;
            background-color: #FF4B4B;
            border: None;
            padding: 0.5em 1.5em;
            border-radius: 0.5em;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# Load model & encoders
model = joblib.load("xgboost_final_model.pkl")
encoder_files = [
    'le_Item_Fat_Content.pkl', 'le_Item_Type.pkl', 'le_Outlet_Identifier.pkl',
    'le_Outlet_Size.pkl', 'le_Outlet_Location_Type.pkl', 'le_Outlet_Type.pkl'
]
encoders = {file.replace("le_", "").replace(".pkl", ""): joblib.load(file) for file in encoder_files}

# Define feature order
feature_order = [
    'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
    'Item_MRP', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type',
    'Outlet_Type', 'Outlet_Years', 'Item_Category'
]

# Sidebar
page = st.sidebar.radio("📍 Choose a page", ["🧮 Predict Sales", "📊 EDA Dashboard"])

# -------------------- Predict Sales --------------------
if page == "🧮 Predict Sales":
    st.title("🛒 Swiggy Instamart Sales Predictor")

    with st.form("prediction_form"):
        st.subheader("📝 Enter Product & Outlet Info")

        item_weight = st.number_input("1️⃣ Item Weight", min_value=0.0, step=0.1)
        item_fat_content = st.selectbox("2️⃣ Item Fat Content", ['Low Fat', 'Regular'])
        item_visibility = st.number_input("3️⃣ Item Visibility", min_value=0.0, step=0.01)
        item_type = st.selectbox("4️⃣ Item Type", [
            'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
            'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
            'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods',
            'Others', 'Seafood'])
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

        df = df[feature_order]  # Ensure correct order

        pred_log = model.predict(df)
        pred = np.expm1(pred_log[0])

        st.success(f"💰 **Predicted Sales: ₹{round(pred, 2)}**")

        # Download result
        csv = df.copy()
        csv['Predicted_Sales'] = [round(pred, 2)]
        csv_bytes = csv.to_csv(index=False).encode()
        st.download_button("⬇ Download Prediction", data=csv_bytes, file_name="predicted_sales.csv", mime="text/csv")

# -------------------- EDA Dashboard --------------------
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

        st.subheader("📌 Pairplot (Numeric Features)")
        if len(num_cols) >= 2:
            fig = px.scatter_matrix(df[num_cols])
            st.plotly_chart(fig)

        st.subheader("🧮 Value Counts (Categorical Columns)")
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        if cat_cols:
            cat = st.selectbox("Select categorical column", cat_cols)
            fig = px.bar(df[cat].value_counts().reset_index(), x='index', y=cat, labels={'index': cat, cat: 'Count'})
            st.plotly_chart(fig)
