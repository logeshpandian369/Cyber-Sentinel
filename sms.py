import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from twilio.rest import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# TWILIO CONFIG FROM .env
TWILIO_SID = os.getenv('TWILIO_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')
RECEIVER_PHONE_NUMBER = os.getenv('RECEIVER_PHONE_NUMBER')

def send_sms_alert(report_text):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"🚨 Intrusion Detected!\n\n{report_text[:300]}...",
            from_=TWILIO_PHONE_NUMBER,
            to=RECEIVER_PHONE_NUMBER
        )
        print(f"📱 SMS alert sent! SID: {message.sid}")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        print("✅ Data loaded successfully!")

        categorical_cols = ['proto', 'service', 'state']
        if 'attack_cat' not in df.columns:
            raise ValueError("attack_cat column is missing!")

        df = df[df['attack_cat'].notna()]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def train_and_evaluate_model(df):
    if df is not None and not df.empty:
        X = df.drop(columns=['attack_cat'])
        y = df['attack_cat']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print("📊 Model Accuracy:", accuracy)
        print("🎯 Precision:", precision)
        print("🔍 Recall:", recall)
        print("🏅 F1 Score:", f1)

        report = classification_report(y_test, y_pred)
        print("📄 Classification Report:\n", report)

        # Save report to file
        report_file_path = 'classification_report.txt'
        with open(report_file_path, 'w') as f:
            f.write(report)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        # Send SMS if attack detected
        if any(pred != 'Normal' and pred.lower() != 'benign' for pred in y_pred):
            send_sms_alert(report)
        else:
            print("✅ No attacks detected. SMS not sent.")
    else:
        print("⚠️ No data available for training.")

# Load and run
file_path = r'E:\Downloads\IoT-Devices-Intrusion-Detection-main\IoT-Devices-Intrusion-Detection-main\UNSW_NB15_testing-set.parquet'
data = load_data(file_path)
train_and_evaluate_model(data)
