import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class DoSDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DoS Intrusion Detection System")
        self.model = None
        self.df = None
        self.result_box = scrolledtext.ScrolledText(root, height=15, width=80)
        self.result_box.pack(pady=10)

        tk.Button(root, text="Load and Preprocess Dataset", command=self.load_and_preprocess).pack(pady=5)
        tk.Button(root, text="Run Detection", command=self.run_detection).pack(pady=5)
        tk.Button(root, text="Show Confusion Matrix", command=self.show_confusion_matrix).pack(pady=5)

        self.conf_matrix = None
        self.metrics = {}

    def log(self, message):
        self.result_box.insert(tk.END, message + "\n")
        self.result_box.see(tk.END)

    def encode_text_dummy(self, df, name):
        if name in df.columns:
            dummies = pd.get_dummies(df[name], prefix=name)
            df = pd.concat([df.drop(name, axis=1), dummies], axis=1)
        return df

    def load_and_preprocess(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        self.log(f"Loading dataset: {file_path}")
        df = pd.read_csv(file_path, encoding='latin1', low_memory=False)

        drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
                        "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
                        "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
                        "tcp.dstport", "udp.port", "mqtt.msg"]
        df.drop(columns=drop_columns, inplace=True, errors='ignore')
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df = shuffle(df)
        df = df[~df['Attack_type'].str.startswith('DDoS_')]

        for col in ['http.request.method', 'http.referer', 'http.request.version',
                    'dns.qry.name.len', 'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic']:
            df = self.encode_text_dummy(df, col)

        self.df = df.copy()
        self.log("Preprocessing completed.\nFiltered Attack Types:\n" + str(df['Attack_type'].value_counts()))

    def run_detection(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please load and preprocess the dataset first.")
            return

        df = self.df.copy()
        label_col = "Attack_type"
        X = df.drop(columns=[label_col])
        y = df[label_col]

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=1, stratify=y_encoded)

        num_classes = len(np.unique(y_train))
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        self.model = load_model('model.h5')
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test_cat, axis=1)

        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        self.conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

        self.metrics = {"Precision": precision, "Recall": recall, "F1-Score": f1}
        self.log(f"Precision: {precision:.4f}")
        self.log(f"Recall: {recall:.4f}")
        self.log(f"F1-score: {f1:.4f}")
        self.log("Detection complete.")
        self.show_metric_chart()

    def show_confusion_matrix(self):
        if self.conf_matrix is None:
            messagebox.showwarning("Warning", "Please run detection first.")
            return
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    def show_metric_chart(self):
        if not self.metrics:
            return

        metric_window = tk.Toplevel(self.root)
        metric_window.title("Model Performance Metrics")

        fig, ax = plt.subplots(figsize=(5, 4))
        names = list(self.metrics.keys())
        values = list(self.metrics.values())
        ax.bar(names, values, color=['#3498db', '#2ecc71', '#e74c3c'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Precision, Recall, F1-Score")

        canvas = FigureCanvasTkAgg(fig, master=metric_window)
        canvas.draw()
        canvas.get_tk_widget().pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = DoSDetectionApp(root)
    root.mainloop()
