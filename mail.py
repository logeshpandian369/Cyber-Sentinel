import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import smtplib
from email.message import EmailMessage
import datetime as pd

# --- Load and prepare your dataset ---
data = pd.read_csv('E:/Downloads/IoT-Devices-Intrusion-Detection-main/IoT-Devices-Intrusion-Detection-main/iot23_combined.csv')  # Replace with your dataset file
X = data.drop(columns=['label_column'])  # Replace with actual features
y = data['label_column']  # Replace with actual labels if needed

# Split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Anomaly Detection to Identify Undetected Attacks ---
print("\n🔍 Running anomaly detection using Isolation Forest...")

iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
anomaly_preds = iso_forest.fit_predict(X_test.values)  # Ensure X_test is a NumPy array

# Anomalies are labeled -1
anomaly_indices = np.where(anomaly_preds == -1)[0]
num_anomalies = len(anomaly_indices)
print(f"🚨 Detected {num_anomalies} potential new/unknown attack(s) using anomaly detection.")

# --- Email Alert (trigger if anomalies detected) ---
if num_anomalies > 0:
    sender_email = "12csgowtham@gmail.com"
    sender_password = "qotd looy mbsz nakp"  # Use App Password if using Gmail
    recipient_email = "lkumar4105@gmail.com"

    msg = EmailMessage()
    msg['Subject'] = '🚨 IoT Intrusion Detection Alert'
    msg['From'] = sender_email
    msg['To'] = recipient_email

    alert_message = f"""
    ALERT: {num_anomalies} potential new or undetected IoT attacks were found!
    Please check the intrusion detection system logs for further details.

    Time: {pd.datetime.now()}
    """
    msg.set_content(alert_message)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:  # or your provider's SMTP server
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        print("✅ Alert email sent successfully.")
    except Exception as e:
        print("❌ Failed to send email:", e)
else:
    print("✅ No anomalies detected. System appears normal.")
#E:/Downloads/IoT-Devices-Intrusion-Detection-main/IoT-Devices-Intrusion-Detection-main/DNN-EdgeIIoT-dataset.csv
#E:/Downloads/IoT-Devices-Intrusion-Detection-main/IoT-Devices-Intrusion-Detection-main/iot23_combined.csv
