import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import base64

# Set page config and background
st.set_page_config(page_title="Swiggy Sales App", layout="wide")

# --- Background image via base64 ---
def set_bg():
    bg_img = """
    https://images.unsplash.com/photo-1606813902883-ade4bd6d07da?auto=format&fit=crop&w=1470&q=80
    """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_img}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# Load Model & Encoders
model = joblib.load("xgboost_final_model.pkl")
encoder_files = [
    'le_Item_Fat_Content.pkl',
    'le_Item_Type.pkl',
    'le_Outlet_Identifier.pkl',
    'le_Outlet_Size.pkl',
    'le_Outlet_Location_Type.pkl',
    'le_Outlet_Type.pkl'
]
encoders = {file.replace("le_", "").replace(".pkl", ""): joblib.load(file) for file in encoder_files}

# Sidebar navigation
page = st.sidebar.selectbox("Choose an option", ["🧮 Predict Sales", "📊 EDA Dashboard"])

# ----------------------------
# 🧮 PREDICTION TAB
# ----------------------------
if page == "🧮 Predict Sales":
    st.title("🛒 Swiggy Instamart Sales Predictor")
    st.write("Predict sales based on item and outlet details.")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            item_weight = st.number_input("Item Weight", min_value=0.0, step=0.1)
            item_visibility = st.number_input("Item Visibility", min_value=0.0, step=0.01)
            item_mrp = st.number_input("Item MRP", min_value=0.0, step=1.0)
            outlet_years = st.number_input("Years Since Outlet Established", min_value=0, step=1)
            item_category = st.selectbox("Item Category Code", [0, 1, 2])  # DR=0, FD=1, NC=2

        with col2:
            item_fat_content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
            item_type = st.selectbox("Item Type", [
                'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
                'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
                'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods',
                'Others', 'Seafood'
            ])
            outlet_id = st.selectbox("Outlet Identifier", [
                'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
                'OUT045', 'OUT017', 'OUT046', 'OUT035', 'OUT019'
            ])
            outlet_size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
            outlet_loc = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
            outlet_type = st.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_dict = {
            'Item_Weight': item_weight,
            'Item_Visibility': item_visibility,
            'Item_MRP': item_mrp,
            'Outlet_Years': outlet_years,
            'Item_Fat_Content': item_fat_content,
            'Item_Type': item_type,
            'Outlet_Identifier': outlet_id,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_loc,
            'Outlet_Type': outlet_type,
            'Item_Category': item_category
        }

        df = pd.DataFrame([input_dict])

        for col in encoders:
            df[col] = encoders[col].transform(df[col])

        pred_log = model.predict(df)
        pred = np.expm1(pred_log[0])
        st.success(f"💰 **Predicted Sales: ₹{round(pred, 2)}**")

# ----------------------------
# 📊 EDA TAB
# ----------------------------
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis Dashboard")
    uploaded_file = st.file_uploader("📁 Upload a CSV file for EDA", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("🔍 Data Preview")
        st.dataframe(df.head())

        st.subheader("📊 Basic Statistics")
        st.write(df.describe())

        st.subheader("🎯 Column Distribution")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if numeric_cols:
            selected_col = st.selectbox("Choose a column to plot distribution", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_col], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found.")

        st.subheader("📈 Correlation Heatmap")
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")
