import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import smtplib
import ssl
from email.message import EmailMessage
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# CONFIGURE EMAIL SETTINGS
EMAIL_SENDER = '12csgowtham@gmail.com'
EMAIL_PASSWORD = 'qlwt ryvi gjdo orxj'  # Use app password if using Gmail with 2FA
EMAIL_RECEIVER = '12csgowtham@gmail.com'

def send_email_alert(report_text, report_file_path):
    try:
        msg = EmailMessage()
        msg['Subject'] = '🚨 Intrusion Detection Alert!'
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg.set_content(f"Intrusions have been detected in the dataset.\n\nClassification Summary:\n\n{report_text}")

        with open(report_file_path, 'rb') as f:
            report_data = f.read()
            msg.add_attachment(report_data, maintype='application', subtype='octet-stream', filename='classification_report.txt')

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print("📧 Alert email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        print("Data loaded successfully!")
        categorical_cols = ['proto', 'service', 'state']
        if 'attack_cat' not in df.columns:
            raise ValueError("attack_cat column is missing!")

        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
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

        print("Model Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        report = classification_report(y_test, y_pred)
        print("Classification Report:\n", report)

        # Save report to file
        report_file_path = 'classification_report.txt'
        with open(report_file_path, 'w') as f:
            f.write(report)

        # Visualize Confusion Matrix
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.show()

        # Send Email if attack detected
        if any(pred != 'Normal' and pred.lower() != 'benign' for pred in y_pred):
            send_email_alert(report, report_file_path)
        else:
            print("✅ No attacks detected. Email not sent.")
    else:
        print("No data available for training.")

# Load and run
file_path = r'E:\Downloads\IoT-Devices-Intrusion-Detection-main\IoT-Devices-Intrusion-Detection-main\UNSW_NB15_testing-set.parquet'
data = load_data(file_path)
train_and_evaluate_model(data)
