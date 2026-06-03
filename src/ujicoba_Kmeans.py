# %%
import pandas as pd
import joblib

# 1. Memuat (Load) Model dan Scaler yang sudah kamu simpan tadi
# Ibaratnya kita menghidupkan kembali AI yang sudah pintar
loaded_scaler = joblib.load('scaler_finesse.pkl')
loaded_kmeans = joblib.load('kmeans_finesse.pkl')

# 2. Simulasi Data User Baru (Skenario: Mahasiswa bernama Budi)
# Misal: Budi punya jatah bulanan Rp2.000.000
# Tapi dia sering jajan kopi/boba, total transaksi 28 kali.
# Uangnya sudah habis Rp3.500.000 (over budget).
data_baru = {
    'total_spent': [3500000],
    'transaction_count': [28],
    'monthly_budget': [2000000]
}
df_new_user = pd.DataFrame(data_baru)

# 3. Feature Engineering (Sama persis seperti saat training)
# AI kita butuh kolom tambahan ini untuk bekerja
df_new_user['budget_utilization'] = df_new_user['total_spent'] / df_new_user['monthly_budget']

# 4. Susun urutan kolom sesuai urutan training
features_to_predict = df_new_user[['total_spent', 'transaction_count', 'budget_utilization']]

# 5. Data Scaling (SANGAT KRUSIAL)
# Perhatikan: Kita pakai .transform(), BUKAN .fit_transform()
# Karena kita mau menyesuaikan data Budi dengan standar skala yang lama
new_user_scaled = loaded_scaler.transform(features_to_predict)

# 6. Minta Model K-Means Menebak Liganya!
prediksi_cluster = loaded_kmeans.predict(new_user_scaled)[0]

# --- Menggabungkan dengan Fungsi Logika Misi yang tadi ---
def get_adaptive_mission(cluster_id):
    if cluster_id == 0: return "Gold", "Pertahankan prestasimu! Tabung 10% sisa uangmu."
    elif cluster_id == 1: return "Silver", "Kamu sering jajan impulsif! Batasi 1 transaksi/hari."
    elif cluster_id == 2: return "Bronze", "Awas pengeluaran besar! Tunda beli barang non-esensial."
    elif cluster_id == 3: return "Iron", "🚨 DARURAT! Jangan jajan di luar 5 hari ke depan!"
    else: return "Unranked", "Catat transaksimu!"

nama_liga, teks_misi = get_adaptive_mission(prediksi_cluster)

# Tampilkan Hasil Akhir di layar (seolah-olah ini layar HP Budi)
print("=== HASIL ANALISIS FINESSE ===")
print(f"Pemakaian Budget : {df_new_user['budget_utilization'][0]*100:.1f}%")
print(f"Prediksi AI      : Masuk Cluster {prediksi_cluster}")
print(f"Penempatan Liga  : {nama_liga}")
print(f"Misi Otomatis    : {teks_misi}")
# %%
