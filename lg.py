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
from fpdf import FPDF
import os

# Email config
EMAIL_SENDER = '12csgowtham@gmail.com'
EMAIL_PASSWORD = 'moej xpeq jpsc wxyu'
EMAIL_RECEIVER = 'veerakaleeswaran2k3@gmail.com'

def send_email_alert(subject, body, attachments=None):
    try:
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg.set_content(body)

        if attachments:
            for file_path in attachments:
                with open(file_path, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(file_path)
                    msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print("📧 Email sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def generate_pdf_report(report_text, f1_score_value):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Intrusion Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"F1 Score: {f1_score_value:.2f}", ln=True)
    pdf.ln(10)

    # Clean the report text by removing unsupported characters (like emojis)
    clean_text = report_text.encode('latin-1', 'ignore').decode('latin-1')

    for line in clean_text.split('\n'):
        pdf.multi_cell(0, 10, txt=line)

    pdf_file_path = "classification_report.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path


def save_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    image_path = "confusion_matrix.png"
    plt.savefig(image_path)
    plt.close()
    return image_path

def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        print("✅ Data loaded successfully!")

        categorical_cols = ['proto', 'service', 'state']
        if 'attack_cat' not in df.columns:
            raise ValueError("⚠️ 'attack_cat' column is missing in the dataset!")

        df = df[df['attack_cat'].notna()]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df.dropna(inplace=True)

        # Send notification email
        send_email_alert(
            "✅ Data Load Notification",
            "The dataset was successfully loaded and processed."
        )

        return df
    except Exception as e:
        error_msg = f"❌ Error loading data: {e}"
        print(error_msg)
        send_email_alert("❌ Data Load Failed", error_msg)
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

        print(f"📊 Accuracy: {accuracy:.4f} | 🎯 Precision: {precision:.4f} | 🔍 Recall: {recall:.4f} | 🏅 F1 Score: {f1:.4f}")

        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

        # Save report and confusion matrix
        pdf_path = generate_pdf_report(report, f1)
        cm_image_path = save_confusion_matrix(cm, model.classes_)

        if any(pred != 'Normal' and pred.lower() != 'benign' for pred in y_pred):
            send_email_alert(
                "🚨 Intrusion Detection Alert!",
                f"Suspicious activity detected!\n\nF1 Score: {f1:.4f}",
                attachments=[pdf_path, cm_image_path]
            )
        else:
            print("✅ No attacks detected. Email not sent.")
    else:
        print("⚠️ No data available for training.")

# Run pipeline
file_path = r'E:\Downloads\IoT-Devices-Intrusion-Detection-main\IoT-Devices-Intrusion-Detection-main\UNSW_NB15_testing-set.parquet'
data = load_data(file_path)
train_and_evaluate_model(data)
