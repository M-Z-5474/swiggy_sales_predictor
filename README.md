![GitHub last commit](https://img.shields.io/github/last-commit/M-Z-5474/swiggy_sales_predictor)
![GitHub repo size](https://img.shields.io/github/repo-size/M-Z-5474/swiggy_sales_predictor)
![Python version](https://img.shields.io/badge/Python-3.10-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-orange?logo=streamlit)

[![Open in Colab](https://img.shields.io/badge/Open%20in-Google%20Colab-yellow?logo=googlecolab)](https://colab.research.google.com/drive/your-colab-id-here)

# 🛒 Swiggy Instamart-Style Sales Prediction using Machine Learning

[![Streamlit](https://img.shields.io/badge/Deployed%20on-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://swiggysalespredictor-58tlpumxyim6ux5a4oiikh.streamlit.app/)

A machine learning web app that predicts **item outlet sales** based on historical product and outlet data. Built using **XGBoost**, trained on the BigMart dataset, and deployed with **Streamlit Cloud**.

---

## 📌 Features

- Predicts item sales using input product and outlet details.
- Interactive UI built with Streamlit.
- Trained using advanced regression models (XGBoost, Random Forest, etc.).
- Model evaluation with RMSE & R² score comparison.
- Encoded inputs using `LabelEncoder`.

---

## 🚀 Live Demo

👉 Click on it for live demo: https://swiggysalespredictor-58tlpumxyim6ux5a4oiikh.streamlit.app

---

## 📁 Project Structure

```

swiggy_sales_predictor/
│
├── 📁 data/
│ ├── train.csv # Training data
│ ├── test.csv # Test data
│ └── final_predictions.xlsx # Model predictions
│
├── 📁 notebook/
│ ├── Swiggy_Instamart_Sales_Prediction.ipynb # Full ML pipeline
│ └── README.md # Notebook overview
|
├── app.py                   # Streamlit app
├── requirements.txt         # All necessary Python packages
├── xgboost\_final\_model.pkl  # Trained XGBoost model
├── le\_Item\_Fat\_Content.pkl  # Label encoders for categorical features
├── le\_Item\_Type.pkl
├── le\_Outlet\_Identifier.pkl
├── le\_Outlet\_Size.pkl
├── le\_Outlet\_Location\_Type.pkl
├── le\_Outlet\_Type.pkl
└── README.md                # Project documentation

````

---

## ⚙️ Setup Instructions (For Local Development)

1. Clone the repository:
```bash
git clone https://github.com/your-username/swiggy_sales_predictor.git
cd swiggy_sales_predictor
````

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 📧 Contact

**🙋‍♂️ Author Muhammad Zain Mushtaq**

🔗 GitHub: https://github.com/M-Z-5474

📧 Email: m.zainmushtaq74@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/muhammad-zain-m-a75163358/

---
## 🌟 Star This Repo!
## If you found this helpful, feel free to ⭐ this project. It motivates further contributions.
Thank you for visiting! 🙌


---


