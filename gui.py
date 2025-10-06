import tkinter as tk
from tkinter import filedialog, messagebox
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

# Email configuration
EMAIL_SENDER = 'lk15211009@gmail.com'
EMAIL_PASSWORD = 'bykb mmgn dttf gmsw'
EMAIL_RECEIVER = 'lizzy0036vm@gmail.com'

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
        return True
    except Exception as e:
        print(f"❌ Email Error: {e}")
        return False

def generate_pdf_report(report_text, f1_score_value):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Intrusion Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"F1 Score: {f1_score_value:.2f}", ln=True)
    pdf.ln(10)
    clean_text = report_text.encode('latin-1', 'ignore').decode('latin-1')
    for line in clean_text.split('\n'):
        pdf.multi_cell(0, 10, txt=line)
    path = "classification_report.pdf"
    pdf.output(path)
    return path

def save_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    path = "confusion_matrix.png"
    plt.savefig(path)
    plt.close()
    return path

def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        categorical_cols = ['proto', 'service', 'state']
        if 'attack_cat' not in df.columns:
            raise ValueError("'attack_cat' column missing in dataset")
        df = df[df['attack_cat'].notna()]
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        df.dropna(inplace=True)
        send_email_alert("✅ Data Load Notification", "Dataset loaded and processed successfully.")
        return df
    except Exception as e:
        send_email_alert("❌ Data Load Failed", f"Error: {e}")
        raise

def train_and_evaluate_model(df):
    X = df.drop(columns=['attack_cat'])
    y = df['attack_cat']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    pdf_path = generate_pdf_report(report, f1)
    cm_img = save_confusion_matrix(cm, model.classes_)
    attack_counts = pd.Series(y_pred).value_counts()
    sent = send_email_alert("🚨 Intrusion Detection Report", f"F1 Score: {f1:.4f}\nReport Attached", [pdf_path, cm_img])
    return f1, report, cm_img, pdf_path, attack_counts, sent

# ================= GUI =================

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("Cyber Sentinel - IoT Attack Detection")
        self.master.geometry("700x600")

        self.file_path = None

        tk.Label(master, text="Cyber Sentinel", font=("Arial", 18, "bold")).pack(pady=10)

        self.select_button = tk.Button(master, text="Select Parquet File", command=self.select_file)
        self.select_button.pack(pady=10)

        self.run_button = tk.Button(master, text="Run Detection", command=self.run_pipeline, state='disabled')
        self.run_button.pack(pady=10)

        self.status_label = tk.Label(master, text="", fg="blue")
        self.status_label.pack(pady=5)

        self.output_text = tk.Text(master, height=20, width=80)
        self.output_text.pack(pady=10)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Parquet files", "*.parquet")])
        if path:
            self.file_path = path
            self.status_label.config(text=f"Selected file: {os.path.basename(path)}")
            self.run_button.config(state='normal')

    def run_pipeline(self):
        self.output_text.delete("1.0", tk.END)
        try:
            df = load_data(self.file_path)
            f1, report, cm_img, pdf_path, attack_counts, sent = train_and_evaluate_model(df)

            self.output_text.insert(tk.END, f"📊 F1 Score: {f1:.4f}\n\n")
            self.output_text.insert(tk.END, "📄 Classification Report:\n")
            self.output_text.insert(tk.END, report + "\n\n")
            self.output_text.insert(tk.END, "🔥 Attack Distribution:\n")
            self.output_text.insert(tk.END, attack_counts.to_string() + "\n\n")

            # Plot attack distribution
            attack_counts.plot(kind='bar', title='Detected Attack Counts')
            plt.tight_layout()
            plt.savefig("attack_distribution.png")
            plt.close()

            self.output_text.insert(tk.END, f"📧 Email sent: {'Yes' if sent else 'No'}\n")
            messagebox.showinfo("Done", "Detection completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Pipeline failed:\n{e}")
            self.status_label.config(text="Error occurred.")

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
