import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score

# Print feature abbreviations
def print_abbreviations():
    print("""
Dataset Feature Abbreviations:
- dur: duration of the network flow (seconds)
- proto: protocol type (tcp, udp, icmp)
- service: network service type (http, dns, etc.)
- state: connection state (ESTABLISHED, etc.)
- spkts: source to destination packet count
- dpkts: destination to source packet count
- sbytes: source to destination byte count
- dbytes: destination to source byte count
- rate: packets per second
- sload: source bits per second
- dload: destination bits per second
- sloss: source packet loss count
- dloss: destination packet loss count
- sinpkt: source inbound packets
- dinpkt: destination inbound packets
- sjit: source jitter (variation in packet delay)
- djit: destination jitter
- swin: source TCP window size
- stcpb: source TCP base sequence number
- dtcpb: destination TCP base sequence number
- dwin: destination TCP window size
- tcprtt: TCP round trip time
- synack: time between SYN and ACK packets
- ackdat: time between ACK and data packets
- smean: mean packet size from source
- dmean: mean packet size from destination
- trans_depth: HTTP transaction depth
- response_body_len: length of HTTP response body
- ct_src_dport_ltm: count of flows with same source/destination ports (last 2 sec)
- ct_dst_sport_ltm: count of flows with same destination/source ports (last 2 sec)
- is_ftp_login: FTP login attempt flag (0 or 1)
- ct_ftp_cmd: count of FTP commands
- ct_flw_http_mthd: count of HTTP methods
- is_sm_ips_ports: suspicious IPs and ports flag (0 or 1)
- attack_cat: attack category label
- label: 0 = normal, 1 = attack
""")

# Load and preprocess dataset
def load_data(file_path):
    try:
        df = pd.read_parquet(file_path)
        print("Data loaded successfully!")
        print(df.head())  # Check if data is loaded
        print(df.dtypes)  # Check data types

        # Convert categorical columns to one-hot encoding
        categorical_cols = ['proto', 'service', 'state', 'attack_cat']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to test 3D logic (dummy example)
def split_3d_and_label(arr, time, window_size):
    try:
        if len(arr) < 2:
            print("Insufficient data in array, skipping...")
            return None, None

        low, middle = 0, min(window_size // 2, len(arr) - 1)

        if not isinstance(arr[low][1], (int, float, str)) or not isinstance(arr[middle][1], (int, float, str)):
            print("Invalid data format, skipping...")
            return None, None

        try:
            val1 = float(arr[low][1])
            val2 = float(arr[middle][1])
        except ValueError:
            print(f"Skipping non-numeric values: {arr[low][1]}, {arr[middle][1]}")
            return None, None

        if int(abs(val1 - val2)) < time:
            print("Condition met, processing data...")
            return val1, val2
        else:
            print("Condition not met, skipping...")
            return None, None
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

# Visualize dataset
def visualize_data(df):
    if df is not None and not df.empty:
        print("Data Summary:\n", df.describe())
        print("Columns in dataset:", df.columns)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols].dropna()

        if len(numeric_cols) > 0:
            df.hist(bins=50, figsize=(15, 10))
            plt.suptitle("Histogram of Features")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
        else:
            print("No numeric columns found for histogram.")

        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 6))
            corr_matrix = df.corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()
            plt.show()
        else:
            print("Not enough numeric columns for heatmap.")
    else:
        print("No data to visualize or DataFrame is empty.")

# Train and evaluate the ML model
def train_and_evaluate_model(df):
    if df is not None and not df.empty:
        df = df.dropna()
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        print("Model Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        print("Classification Report:\n", classification_report(y_test, y_pred))
    else:
        print("No data available for training.")

# Main execution
if __name__ == "__main__":
    print_abbreviations()  # Show feature details first

    file_path = r'E:\Downloads\IoT-Devices-Intrusion-Detection-main\IoT-Devices-Intrusion-Detection-main\UNSW_NB15_testing-set.parquet'
    data = load_data(file_path)

    # Optional: test dummy function
    arr = [[0, '12.34'], [1, 'unas'], [2, '56.78']]
    time = 350
    split_3d_and_label(arr, time, 350)

    visualize_data(data)
    train_and_evaluate_model(data)
