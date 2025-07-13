import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load the model and encoders
# -----------------------------
model = joblib.load("xgboost_final_model.pkl")

encoder_files = [
    'le_Item_Fat_Content.pkl',
    'le_Item_Type.pkl',
    'le_Outlet_Identifier.pkl',
    'le_Outlet_Size.pkl',
    'le_Outlet_Location_Type.pkl',
    'le_Outlet_Type.pkl'
]

encoders = {}
for file in encoder_files:
    col_name = file.replace("le_", "").replace(".pkl", "")
    encoders[col_name] = joblib.load(file)

# ✅ Feature order used during training
feature_order = [
    'Item_Weight',
    'Item_Fat_Content',
    'Item_Visibility',
    'Item_Type',
    'Item_MRP',
    'Outlet_Identifier',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type',
    'Outlet_Years',
    'Item_Category'
]

# -----------------------------
# Streamlit App UI
# -----------------------------
st.set_page_config(page_title="🛒 Swiggy Instamart Sales Predictor", layout="centered")
st.title("🛒 Swiggy Instamart Sales Predictor")
st.write("Predict sales based on item and outlet details.")

# -----------------------------
# Input Form
# -----------------------------
with st.form("input_form"):
    item_weight = st.number_input("Item Weight (e.g., 10.5)", min_value=0.0, step=0.1)
    item_visibility = st.number_input("Item Visibility (e.g., 0.05)", min_value=0.0, step=0.01)
    item_mrp = st.number_input("Item MRP", min_value=0.0, step=1.0)
    outlet_years = st.number_input("Years Since Outlet Established", min_value=0, step=1)

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

    item_category = st.selectbox("Item Category Code", [0, 1, 2])  # DR=0, FD=1, NC=2

    submitted = st.form_submit_button("🔍 Predict")

# -----------------------------
# Predict & Show Result
# -----------------------------
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

    # Apply label encoding
    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    # ✅ Reorder columns to match training feature order
    df = df[feature_order]

    # Predict
    y_pred_log = model.predict(df)
    y_pred = np.expm1(y_pred_log[0])  # Reverse log1p

    st.success(f"💰 **Predicted Sales: ₹{round(y_pred, 2)}**")
