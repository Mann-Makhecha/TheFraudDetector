# Fraud Transaction Detector

🛡️ A Streamlit-powered machine learning application for detecting fraudulent credit card transactions using a simulated dataset of ~1.3 million records.

## Business Problem

Payment fraud causes **massive financial losses globally** — estimated at over $30 billion annually. This application uses supervised machine learning to flag suspicious transactions, helping financial institutions minimise losses and protect customers.

## Features

- **5-Page Interactive Dashboard** built with Streamlit
- **3 ML Models**: Logistic Regression, Decision Tree, Random Forest
- **Class Imbalance Handling** via `class_weight='balanced'`
- **Optimised for Recall** to catch as many fraudulent transactions as possible
- **Live Prediction Interface** with risk scoring

## Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, dataset summary, team info |
| 📊 EDA | 7 interactive visualisations with filters |
| 🤖 Model Training | Train models with tuneable hyperparameters |
| 🔮 Prediction | Real-time fraud prediction with risk score |
| 📊 Comparison | Side-by-side model evaluation & best model selection |

## Dataset

- **Source**: [Kaggle — Credit Card Fraud (Simulated)](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **File**: `fraudTrain.csv` (~1.3M rows)
- **Target**: `is_fraud` (0 = Legitimate, 1 = Fraud)
- **Class Distribution**: ~99.4% Legitimate / ~0.6% Fraud

## Tech Stack

- Python 3.10+
- Streamlit
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

## Project Structure

```
TheFraudDetector/
├── app.py                  # Streamlit entry point
├── pages/
│   ├── 1_Home.py           # Project overview
│   ├── 2_EDA.py            # Exploratory data analysis
│   ├── 3_Model_Training.py # Model training interface
│   ├── 4_Prediction.py     # Live prediction form
│   └── 5_Model_Comparison.py # Model comparison dashboard
├── utils/
│   ├── data_loader.py      # Data loading & feature engineering
│   ├── model_utils.py      # Training, evaluation & persistence
│   └── visualizations.py   # Chart functions
├── models/                 # Saved models (generated at runtime)
├── fraudTrain.csv          # Training dataset
├── fraudTest.csv           # Test dataset
├── requirements.txt        # Python dependencies
└── README.md
```

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ensure fraudTrain.csv is in the project root

# 3. Launch the app
streamlit run app.py
```

The app will open at **http://localhost:8501**.

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `hour` | Hour of transaction (0–23) |
| `day_of_week` | Day of week (0=Mon – 6=Sun) |
| `month` | Month (1–12) |
| `age` | Customer age at transaction time |
| `distance_km` | Haversine distance between cardholder and merchant |
| `amt_log` | Log-transformed transaction amount |
| `category_*` | One-hot encoded merchant category |
| `gender_*` | One-hot encoded gender |

## Model Training

All models use `class_weight='balanced'` to handle the severe class imbalance (~0.6% fraud). The primary optimisation metric is **Recall** — the proportion of actual fraud cases correctly identified.

| Model | Key Hyperparameters |
|-------|-------------------|
| Logistic Regression | C, max_iter |
| Decision Tree | max_depth, min_samples_split |
| Random Forest | n_estimators, max_depth, min_samples_split |

## License

This project is for educational and demonstration purposes.
