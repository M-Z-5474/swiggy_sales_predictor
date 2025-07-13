# 📁 Data Folder

This folder contains the training and testing datasets used in the **Swiggy Sales Prediction** project.

---

## 🔍 Features Description

### 🔢 Numerical Features:
- **Item_Weight**: Weight of the product.
- **Item_Visibility**: Percentage of total display area in the store allocated to the product.
- **Item_MRP**: Maximum Retail Price (list price) of the product.
- **Outlet_Establishment_Year**: Year in which the store was established.
- **Item_Outlet_Sales**: Sales of the product in a particular store (target variable).

### 🔠 Categorical Features:
- **Item_Identifier**: Unique product ID *(dropped during preprocessing)*.
- **Item_Fat_Content**: Indicates whether the product is low fat or regular.
- **Item_Type**: Category/type of the product.
- **Outlet_Identifier**: Unique store ID.
- **Outlet_Size**: Size of the store (Small/Medium/High).
- **Outlet_Location_Type**: City tier where the store is located.
- **Outlet_Type**: Type of outlet (e.g., Grocery Store, Supermarket Type1/2/3).

---

## 📄 Files Included:
- `train.csv`: Original dataset used to train the machine learning model.
- `test.csv`: Dataset used to evaluate the model’s performance.
- `final_predictions.xlsx`: Contains predicted sales results on the test data after training.
