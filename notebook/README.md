# 📓 Notebook Folder

This folder contains the main Google Colab notebook used in the **Swiggy Instamart Sales Prediction** project.

---

## 📘 Notebook

- `Swiggy_Instamart_Sales_Prediction.ipynb`  
  📊 This notebook implements the **complete end-to-end machine learning pipeline**, including:

  - Data loading and cleaning  
  - Exploratory Data Analysis (EDA)  
  - Feature engineering and transformation  
  - Model training and evaluation (XGBoost, etc.)  
  - Final predictions and result export  

The notebook is modular, well-commented, and can be easily reused for similar retail forecasting tasks.

---

## 🛠 Tools & Libraries Used

- Google Colab  
- Python (Pandas, NumPy)  
- Seaborn, Matplotlib, Plotly  
- Scikit-learn, XGBoost  
- Joblib (for saving/loading models)

---

## 🔍 Model Performance Summary

| Model                   | RMSE     | R²   |
|-------------------------|----------|------|
| Linear Regression       | 1277.99  | 0.40 |
| Random Forest           | 1109.66  | 0.55 |
| XGBoost (default)       | 1072.41  | 0.58 |
| CatBoost                | 1060.18  | 0.59 |
| Gradient Boosting       | 1054.31  | 0.59 |
| **Tuned XGBoost**       | **1051.69** | **0.59** |
| Tuned Gradient Boosting | 1056.81  | 0.59 |

---

## ✅ Output

- Trained model files (`.pkl`) used in the Streamlit app  
- Prediction results exported to Excel (`Final_Predictions.xlsx`)


