import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
import lime
import lime.lime_tabular
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data
def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        print("Data loaded successfully!")
        
        categorical_cols = ['proto', 'service', 'state', 'attack_cat']
        
        # Convert categorical columns to strings first (if not already)
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Convert any remaining object columns to numeric
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Train and evaluate the model
def train_and_evaluate_model(df):
    if df is not None and not df.empty:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Ensure all features are numeric
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_test = X_test.apply(pd.to_numeric, errors='coerce')

        # Fixing class imbalance
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("Model Accuracy:", accuracy)
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

        # SHAP Explanation - Use TreeExplainer for RandomForest
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Convert SHAP values to array if needed
        shap.summary_plot(np.array(shap_values), X_test)

        # LIME Explanation
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, mode="classification", feature_names=X_train.columns, class_names=['False', 'True'])
        exp = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
        exp.show_in_notebook()

# Example Usage
file_path = r'E:\Downloads\IoT-Devices-Intrusion-Detection-main\IoT-Devices-Intrusion-Detection-main\UNSW_NB15_testing-set.parquet'
data = load_data(file_path)

if data is not None:
    train_and_evaluate_model(data)
