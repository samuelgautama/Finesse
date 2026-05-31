# %%
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import Callback, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import os

df = pd.read_csv('C:\\Users\\M S I\\Documents\\Autodidak\\DBS dicoding\\CAPSTONE\\finesse_dataset_engineered.csv')

# %% Fitur Engineering preprocessing 
# hapus dan select fitur
X = df.drop(columns=['user_id', 'financial_health_score'])
y = df['financial_health_score']

# Identifikasi fitur numerik dan kategorikal
for col in X.columns:
    if X[col].dtype == 'bool':
        X[col] = X[col].astype(int)


target_scaler = MinMaxScaler() # fitur numerik dan kategorikal jadi 0-1 tuk exp di Node.js nanti
y_scaled = target_scaler.fit_transform(y.values.reshape(-1, 1))

num_features = [
    'amount', 'monthly_budget', 'cumulative_spend', 
    'transaction_to_budget_ratio', 'budget_utilization_ratio', 
    'user_avg_transaction', 'amount_vs_user_avg'
]
pass_features = [col for col in X.columns if col not in num_features]

# menggabungkan scaler tuk fitur numerik dan encoder untuk fitur kategorikal
preprocessor_dnn = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('pass', 'passthrough', pass_features)
    ])

X_processed = preprocessor_dnn.fit_transform(X)

# Ubah format sparse matrix jadi array biasa (TensorFlow) arsitektu
if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

# %% 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_scaled, test_size=0.2, random_state=42
)

# %% Buat costum callback
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        current_mae = logs.get('mae')
        if current_mae is not None and current_mae <= 0.02:
            print(f"\nTarget MAE <= 0.02 dah pass di epoch {epoch+1}! (MAE: {current_mae:.4f})")
            print("Menghentikan training untuk mencegah overfitting.")
            self.model.stop_training = True

# %%
input_layer = layers.Input(shape=(X_train.shape[1],), name="input_features")
x = layers.Dense(64, activation='relu', name="hidden_layer_1")(input_layer)
x = layers.Dense(32, activation='relu', name="hidden_layer_2")(x)
x = layers.Dense(16, activation='relu', name="hidden_layer_3")(x)
output_layer = layers.Dense(1, activation='linear', name="output_layer")(x)

model = Model(inputs=input_layer, outputs=output_layer, name="Finesse_AI_Juri_Harian")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary() # Arsitektur model DNN kita

# %% TensorBoard callback untuk visualisasi di TensorBoard
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

custom_callback = CustomCallback() # inisialisasinya
# %% Train
history = model.fit(
    X_train, y_train, 
    epochs=50, 
    batch_size=32, 
    validation_split=0.2,
    callbacks=[custom_callback, tensorboard_callback],
    verbose=1
)

# %% Evaluasi model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test, verbose=0)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print(f"Test MSE : {test_loss:.5f}")
print(f"Test MAE : {test_mae:.5f} (Syarat Side Quest: <= 0.02)")
print(f"Test R2  : {r2:.4f} (Akurasi: {r2*100:.2f}%)")
print("="*50)

# %% Simpan model dan preprocessor
joblib.dump(preprocessor_dnn, 'preprocessor_dnn.pkl')
joblib.dump(target_scaler, 'target_scaler.pkl') # simpan untuk Node.js nanti
model.save('dnn_finesse.keras')
print("Model dan Preprocessor berhasil diekspor.")

# %% INFERENCE Model test
print("\n=== UJI INFERENCE SEDERHANA ===")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Evaluasi Data Test - MAE Akhir: {mae:.4f}")

# Contoh mengambil 1 baris data acak dari X_test untuk diprediksi
sample_data = X_test[0:1]
prediksi_scaled = model.predict(sample_data, verbose=0)

# Mengembalikan skala 0-1 menjadi 0-100 lagi menggunakan target_scaler
prediksi_asli = target_scaler.inverse_transform(prediksi_scaled)
jawaban_asli = target_scaler.inverse_transform(y_test[0:1].reshape(-1, 1))

print(f"Hasil Prediksi AI : {prediksi_asli[0][0]:.2f}")
print(f"Kunci Jawaban Asli: {jawaban_asli[0][0]:.2f}")

# %%
