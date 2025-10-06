import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import csv

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
from keras.models import Sequential
from keras.layers import Conv1D, AvgPool1D, Flatten, Dense, Dropout, Softmax
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

df = pd.read_csv('E:\Downloads\IoT-Devices-Intrusion-Detection-main\IoT-Devices-Intrusion-Detection-main\DNN-EdgeIIoT-dataset.csv', low_memory=False)

from sklearn.utils import shuffle
drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4",
                "http.file_data","http.request.full_uri","icmp.transmit_timestamp",
                "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",
                "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.drop_duplicates(subset=None, keep="first", inplace=True)
df = shuffle(df)
df.isna().sum()
print(df['Attack_type'].value_counts())

def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')
encode_text_dummy(df,'http.referer')
encode_text_dummy(df,"http.request.version")
encode_text_dummy(df,"dns.qry.name.len")
encode_text_dummy(df,"mqtt.conack.flags")
encode_text_dummy(df,"mqtt.protoname")
encode_text_dummy(df,"mqtt.topic")

# Apply Median Filtering to all features
window_size_median = 3  # Window size for median filtering
for feature in df.columns:
    if feature != 'Attack_type':  # Exclude the label column
        df[feature + '_median_filtered'] = df[feature].rolling(window=window_size_median, center=True).median()

# Apply Standard Deviation-based Filtering to all features
threshold_std = 3  # Threshold for standard deviation-based filtering
for feature in df.columns:
    if feature != 'Attack_type':  # Exclude the label column
        mean = df[feature].mean()
        std = df[feature].std()
        upper_bound = mean + threshold_std * std
        lower_bound = mean - threshold_std * std
        df[feature + '_std_filtered'] = df[feature].apply(lambda x: x if (lower_bound <= x <= upper_bound) else None)

# Drop original features that were filtered
df.drop(df.columns[df.columns.str.endswith('_filtered')], axis=1, inplace=True)

df.to_csv('preprocessed_DNN.csv', encoding='utf-8', index=False)

df = pd.read_csv('E:\Downloads\IoT-Devices-Intrusion-Detection-main\IoT-Devices-Intrusion-Detection-main\DNN-EdgeIIoT-dataset.csv', low_memory=False)
df

df['Attack_type'].value_counts()

df.info()

feat_cols = list(df.columns)
label_col = "Attack_type"

feat_cols.remove(label_col)

empty_cols = [col for col in df.columns if df[col].isnull().all()]
empty_cols

skip_list = ["icmp.unused", "http.tls_port", "dns.qry.type", "mqtt.msg_decoded_as"]

df[skip_list[3]].value_counts()

fig, (ax1, ax2)  = plt.subplots(nrows=1, ncols=2, figsize=(12,8))
explode = list((np.array(list(df[label_col].dropna().value_counts()))/sum(list(df[label_col].dropna().value_counts())))[::-1])[:]
labels = list(df[label_col].dropna().unique())[:]
sizes = df[label_col].value_counts()[:]

ax2.pie(sizes,  explode=explode, startangle=60, labels=labels, autopct='%1.0f%%', pctdistance=0.8)
ax2.add_artist(plt.Circle((0,0),0.4,fc='white'))
sns.countplot(y=label_col, data=df, ax=ax1)
ax1.set_title("Count of each Attack type")
ax2.set_title("Percentage of each Attack type")
plt.show()

X = df.drop([label_col], axis=1)
y = df[label_col]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

del X
del y

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train =  label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

label_encoder.classes_

from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()
X_train =  min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

print(df.columns)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(X_train.shape)
print(X_test.shape)

input_shape = X_train.shape[1:]

print(X_train.shape, X_test.shape)
print(input_shape)

num_classes = len(np.unique(y_train))
num_classes

from  tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print(y_train.shape, y_test.shape)

from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM

model = Sequential()

# Convolutional layers
model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(2))
model.add(Conv1D(64, 3, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128, 3, activation='relu'))
model.add(MaxPooling1D(2))

# Flatten output from Convolutional layers
model.add(Flatten())

# Reshape output to include time step dimension
model.add(Dense(96 * 128))  # 96 is the number of features, 128 is the number of filters in the last Conv1D layer
model.add(Reshape((96, 128)))

# LSTM layers
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32))

# Dense layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()


opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss= tf.keras.metrics.categorical_crossentropy,
                  metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode="min", verbose=1, min_lr=0)
#plotlosses = PlotLossesKeras()
call_backs = [lr_reduce]
EPOCHS = 15
BATCH_SIZE = 256

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    validation_split=0.1,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=call_backs,
                    #class_weight=class_weights,
                    verbose=1)


# Save the trained model
model.save('model.h5')
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


# call_backs = [checkpoint, early_stopping, lr_reduce, plotlosses]

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

print(y_train.shape)

y_hat = model.predict(X_test)
y_hat = np.argmax(y_hat, axis=1)
print(y_hat)

y_true = np.argmax(y_test, axis=1)

from sklearn.metrics import accuracy_score
def print_score(y_pred, y_real, label_encoder):
    print("Accuracy: ", accuracy_score(y_real, y_pred))
print_score(y_hat, y_true, label_encoder)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_true, y_hat)

class_labels = ['Backdoor', 'DDoS_HTTP', 'DDoS_ICMP', 'DDoS_TCP', 'DDoS_UDP',
                'Fingerprinting', 'MITM', 'Normal', 'Password', 'Port_Scanning',
                'Ransomware', 'SQL_injection', 'Uploading',
                'Vulnerability_scanner', 'XSS']

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

normalized_conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(normalized_conf_matrix, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Normalized Confusion Matrix')
plt.show()

precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)

plt.figure(figsize=(8, 6))
sns.heatmap(precision.reshape(1, -1), annot=True, cmap="Blues", fmt=".2f", xticklabels=class_labels)
plt.xlabel('Classes')
plt.ylabel('Precision')
plt.title('Precision by Class')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(recall.reshape(1, -1), annot=True, cmap="Blues", fmt=".2f", xticklabels=class_labels)
plt.xlabel('Classes')
plt.ylabel('Recall')
plt.title('Recall by Class')
plt.show()
