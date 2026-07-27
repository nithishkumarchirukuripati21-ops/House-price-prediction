# 🏡 House Price Prediction

An end-to-end machine learning app that predicts house sale prices from property features, built on the **Ames Housing dataset** using **XGBoost**, explained with **SHAP**, and served through an interactive **Streamlit** dashboard.

🔗 **Live App:** [house-price-prediction-21.streamlit.app](https://house-price-prediction-21.streamlit.app/)

---

## 📸 Screenshots

**Input form — property dimensions & layout**

![Input form](assets/screenshot-input-form.png)

**Prediction result with SHAP feature breakdown**

![Prediction results](assets/screenshot-results.png)

---

## ✨ Features

- **Instant price predictions** from property details (living area, quality, garage, basement, neighborhood, etc.)
- **Interactive UI** with tabs for Dimensions & Layout, Quality & Finishes, and Location & Utilities
- **Quick presets** — one-click "Standard Family Home" and "Luxury Estate" example inputs
- **Explainable predictions** — TreeSHAP feature-impact breakdown showing which factors pushed the price up or down
- **Price/sqft and summary metrics** shown alongside the prediction
- Clean, dark-themed custom UI

---

## 🧠 How It Works

1. A trained **XGBoost regression model** (`models/best_xgboost_model.pkl`) predicts sale price from encoded property features.
2. User inputs are collected via the Streamlit form, mapped to the same categorical encodings used during training, and aligned to the model's expected `feature_columns`.
3. A **SHAP TreeExplainer** (`models/shap_explainer.pkl`) computes per-feature contributions to explain each individual prediction.
4. Results are displayed with a price badge, key metrics, and a SHAP bar chart + table of the top contributing features.

---

## 📁 Project Structure

```
House-price-prediction/
├── .devcontainer/
│   └── devcontainer.json
├── data/
│   ├── data_description.txt
│   ├── preprocessed_housing.csv
│   ├── test.csv
│   └── train.csv
├── models/
│   ├── best_xgboost_model.pkl
│   ├── feature_columns.pkl
│   ├── model_info.pkl
│   └── shap_explainer.pkl
├── notebooks/
│   ├── EDA.ipynb
│   ├── preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── hyperparameter_tuning.ipynb
│   └── shap_analysis.ipynb
├── src/
├── assets/                  # README screenshots
├── app.py                   # Streamlit application
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML / Modeling | XGBoost, scikit-learn |
| Explainability | SHAP |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Streamlit charts |
| App Framework | Streamlit |
| Model Persistence | Joblib |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/nithishkumarchirukuripati21-ops/House-price-prediction.git
cd House-price-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Dataset

The model is trained on the **Ames Housing dataset**, a widely used benchmark dataset for regression tasks, containing detailed residential property features from Ames, Iowa. It includes ~260 features covering dimensions, quality ratings, location, and utilities — a subset of the most predictive features are used as inputs in the app.

---

## 🔮 Example Inputs

The app lets you configure:
- Overall quality, living area, floor areas, bathrooms, fireplaces
- Exterior & kitchen quality, basement exposure, garage type/finish/capacity
- Neighborhood and central air conditioning

...and returns an instant estimated market value plus a SHAP breakdown of which features drove the prediction up or down.

---

## 🧪 Notebooks

The `notebooks/` folder documents the full modeling workflow:
- **EDA.ipynb** — exploratory data analysis on the Ames Housing dataset
- **preprocessing.ipynb** — cleaning, encoding, and feature engineering
- **model_training.ipynb** — training and comparing regression models
- **hyperparameter_tuning.ipynb** — tuning the final XGBoost model
- **shap_analysis.ipynb** — generating and validating SHAP explanations

---

## 👨‍💻 Developer

**Nithish Kumar Chirukuripati**

🔗 [GitHub](https://github.com/nithishkumarchirukuripati21-ops) • 💼 [LinkedIn](https://www.linkedin.com/in/nithish-kumar-chirukuripati-1a0358338/)

---

## 📄 License

This project is currently unlicensed. Feel free to add a license (e.g. MIT) if you'd like others to freely reuse the code.
