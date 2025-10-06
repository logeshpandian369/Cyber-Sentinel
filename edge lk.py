import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Reshape
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Load dataset
df = pd.read_csv("E:/Downloads/IoT-Devices-Intrusion-Detection-main/IoT-Devices-Intrusion-Detection-main/DNN-EdgeIIoT-dataset.csv", low_memory=False)

# Drop irrelevant columns
drop_cols = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4",
             "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
             "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
             "tcp.dstport", "udp.port", "mqtt.msg"]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Filter only DoS attack types and Normal traffic
dos_attacks = ['DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP']
df = df[df['Attack_type'].isin(dos_attacks + ['Normal'])]

# Drop missing and duplicate values
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
cat_cols.remove('Attack_type')  # keep labels separate
df = pd.get_dummies(df, columns=cat_cols)

# Separate features and labels
X = df.drop(columns=['Attack_type'])
y = df['Attack_type']

# Label encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Model input shape
input_shape = X_train.shape[1:]

# CNN-LSTM Model
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(96 * 128))
model.add(Reshape((96, 128)))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

# Compile model
model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# Train model
history = model.fit(X_train, y_train, validation_split=0.1, epochs=15, batch_size=256, callbacks=[lr_reduce])

# Evaluate model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Accuracy
acc = accuracy_score(y_true, y_pred_classes)
print(f"Accuracy: {acc:.4f}")

# Confusion Matrix
labels = label_encoder.classes_
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
