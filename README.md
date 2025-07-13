
# 🛒 Swiggy Instamart-Style Sales Prediction using Machine Learning

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

swiggy\_sales\_predictor/
│
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

Made by **[Your Name](https://www.linkedin.com/in/your-profile/)**
📬 Feel free to connect or ask for collaboration!

```

---


Would you like me to generate the final `requirements.txt` and `app.py` now to complete your deployment package?
```
