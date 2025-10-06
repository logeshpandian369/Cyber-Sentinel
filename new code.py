import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest

# 1. Load the dataset
file_path = "E:/Downloads/IoT-Devices-Intrusion-Detection-main/IoT-Devices-Intrusion-Detection-main/dataset/iot_dataset.csv"
try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Dataset not found. Please check the file path.")
    exit()

# 2. Drop irrelevant columns (optional, if present)
df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], errors='ignore', inplace=True)

# 3. Identify the correct label column
label_column = 'Label' if 'Label' in df.columns else 'label'
if label_column not in df.columns:
    print("❌ 'Label' column not found in the dataset.")
    print("Available columns:", df.columns.tolist())
    exit()

# 4. Encode categorical features
df = pd.get_dummies(df)

# 5. Separate features and labels
X = df.drop(label_column, axis=1)
y = df[label_column]

# 6. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. Handle class imbalance
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_scaled, y)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 9. Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 10. Evaluate the model
y_pred = clf.predict(X_test)
print("\n📊 Random Forest Classification Report:\n")
print(classification_report(y_test, y_pred))
print("📉 Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# --- BONUS: Unsupervised anomaly detection (undetected attacks) ---
print("\n🔍 Detecting potential undetected attacks (anomalies)...")
iso_forest = IsolationForest(contamination=0.01, random_state=42)
anomaly_labels = iso_forest.fit_predict(X_scaled)  # -1 = anomaly

# Attach anomaly results to the original DataFrame
df['anomaly'] = anomaly_labels

# Filter anomalies
anomalies = df[df['anomaly'] == -1]
print(f"\n⚠️ Potential undetected (anomalous) flows: {len(anomalies)}")
print(anomalies[['anomaly'] + list(df.columns[-6:-1])])  # Show last few columns for context
