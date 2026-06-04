# %% Import Libraries
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Suppress TensorFlow logging for a cleaner terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
from dotenv import load_dotenv
from google import genai

# %% 1. Generative AI Configuration
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("API Key belum diset! Pastikan file .env sudah benar.")

# Initialize GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# %% 2. Redefine Custom Layer for Model Loading
class FinesseDenseLayer(tf.keras.layers.Layer):
    """Implementing standard dense layer (W * x + b) manually."""
    def __init__(self, units=32, activation='relu', **kwargs):
        super(FinesseDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='finesse_weight')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True,
                                 name='finesse_bias')

    def call(self, inputs):
        return self.activation(tf.matmul(inputs, self.w) + self.b)

# %% 3. App Initialization & Model Loading
app = FastAPI(
    title="Finesse AI API",
    description="Unified API for Gamified EXP Prediction (DNN), League Profiling (K-Means), and Context-Aware AI Advisor.",
    version="2.5.0" 
)

# Global Variables
model_dnn = None
preprocessor_dnn = None
target_scaler = None

scaler_km = None
kmeans_model = None
league_mapping = {}
missions_mapping = {}

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

try:
    print("\n⏳ Memuat seluruh arsitektur Machine Learning & Deep Learning...")
    
    # DL Paths
    dnn_dir = PROJECT_ROOT / 'saved_models' / 'Deep_Learning'
    model_dnn = tf.keras.models.load_model(
        dnn_dir / 'finesse_dnn_v1.keras', 
        custom_objects={'FinesseDenseLayer': FinesseDenseLayer}
    )
    preprocessor_dnn = joblib.load(dnn_dir / 'preprocessor_dnn.pkl')
    target_scaler = joblib.load(dnn_dir / 'target_scaler.pkl')
    
    # ML Paths
    km_dir = PROJECT_ROOT / 'saved_models' / 'Machine_Learning'
    scaler_km = joblib.load(km_dir / 'scaler_finesse.pkl')
    kmeans_model = joblib.load(km_dir / 'kmeans_finesse.pkl')
    mappings = joblib.load(km_dir / 'league_mapping.pkl')
    
    league_mapping = mappings['leagues']
    missions_mapping = mappings['missions']
    
    print("✅ Seluruh model (DNN & K-Means) siap melayani request!\n")
except Exception as e:
    print(f"\n❌ GAGAL MEMUAT MODEL: {e}")

# %% 4. Input Schemas (Validasi Data & Template Swagger)
class UnifiedRequest(BaseModel):
    """Skema utama yang menggabungkan seluruh fitur yang dibutuhkan DNN dan K-Means"""
    features: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "amount": 50000,
                    "monthly_budget": 3000000,
                    "cumulative_spend": 1500000,
                    "transaction_count": 28,
                    "transaction_to_budget_ratio": 0.016,
                    "budget_utilization_ratio": 0.5,
                    "user_avg_transaction": 45000,
                    "amount_vs_user_avg": 1.1,
                    "day_of_week": 3,
                    "is_weekend": 0,
                    "is_month_end": 0,
                    "category_Hiburan & Nongkrong": 0,
                    "category_Makan & Minum": 1,
                    "category_Transportasi": 0,
                    "category_Kebutuhan Kuliah": 0,
                    "category_Tagihan & Kos": 0,
                    "payment_method_E-Wallet": 1,
                    "payment_method_Credit Card": 0
                }
            }
        }

class GamificationOnlyRequest(BaseModel):
    """Skema ringan khusus untuk update leaderboard"""
    total_spent: float
    transaction_count: int
    monthly_budget: float

    class Config:
        json_schema_extra = {
            "example": {
                "total_spent": 1500000.0,
                "transaction_count": 28,
                "monthly_budget": 3000000.0
            }
        }

# %% 5. ENDPOINT UTAMA: Dashboard Analysis (All-in-One)
@app.post("/dashboard-analyze")
async def analyze_dashboard(request: UnifiedRequest):
    if any(m is None for m in [model_dnn, preprocessor_dnn, target_scaler, scaler_km, kmeans_model]):
        raise HTTPException(status_code=500, detail="Sistem AI belum siap.")

    try:
        features_dict = request.features
        
        # --- A. GAMIFICATION PHASE (K-Means League Placement) ---
        total_spent = features_dict.get('cumulative_spend', 0)
        trx_count = features_dict.get('transaction_count', 1) 
        monthly_budget = features_dict.get('monthly_budget', 1) 
        
        # Proteksi: Mencegah pembagian dengan nol
        if monthly_budget <= 0:
            monthly_budget = 1 
            
        budget_utilization = total_spent / monthly_budget
        
        km_features = pd.DataFrame({
            'total_spent': [total_spent],
            'transaction_count': [trx_count],
            'budget_utilization': [budget_utilization]
        })
        
        km_scaled = scaler_km.transform(km_features)
        cluster_id = kmeans_model.predict(km_scaled)[0]
        
        user_league = league_mapping.get(cluster_id, "Unranked")
        user_mission = missions_mapping.get(cluster_id, "Catat transaksimu dengan baik!")

        # --- B. DEEP LEARNING PHASE (DNN EXP Prediction) ---
        df_input_dnn = pd.DataFrame([features_dict])
        for col in df_input_dnn.columns:
            if df_input_dnn[col].dtype == 'bool':
                df_input_dnn[col] = df_input_dnn[col].astype(int)
                
        X_processed = preprocessor_dnn.transform(df_input_dnn)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        scaled_prediction = model_dnn.predict(X_processed, verbose=0)
        true_prediction = target_scaler.inverse_transform(scaled_prediction)
        
        # KONVERSI KE EXP: Membulatkan hasil ke bilangan bulat murni (Integer)
        exp_earned = int(round(float(true_prediction[0][0]), 0))

        # --- C. GENERATIVE AI PHASE (Khusus Misi Personal) ---
        kategori_aktif = "Lainnya"
        for key, value in features_dict.items():
            if key.startswith("category_") and value == 1:
                kategori_aktif = key.replace("category_", "")
                break
                
        status_weekend = "Ya" if features_dict.get('is_weekend', 0) == 1 else "Tidak"
        status_akhir_bulan = "Ya" if features_dict.get('is_month_end', 0) == 1 else "Tidak"
        sisa_anggaran = monthly_budget - features_dict.get('cumulative_spend', 0)

        prompt = f"""
        Kamu adalah sistem AI gamifikasi untuk aplikasi keuangan Finesse.
        
        Data transaksi pengguna saat ini:
        - Kategori Transaksi: {kategori_aktif}
        - Nominal: Rp {features_dict.get('amount', 0)}
        - Sisa Anggaran Bulanan: Rp {sisa_anggaran}
        - Liga Gamifikasi Saat Ini: {user_league}
        - Konteks: Akhir Pekan? {status_weekend} | Akhir Bulan? {status_akhir_bulan}
        
        TUGASMU:
        Buat 1 kalimat tantangan/misi spesifik (maksimal 15 kata) untuk beberapa hari ke depan berdasarkan pengeluaran terakhir ini, agar pengguna tetap hemat dan bertahan di Liga {user_league}. Gaya bahasa santai anak muda.
        
        KEMBALIKAN HANYA KALIMAT MISINYA SAJA TANPA EMBEL-EMBEL APAPUN.
        """
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            # Mengambil teks murni dari Gemini sebagai Misi
            dynamic_mission = response.text.strip()
            
        except Exception as e_genai:
            print(f"Error GenAI: {e_genai}")
            # Fallback ke misi bawaan K-Means jika Gemini sedang gangguan
            dynamic_mission = user_mission 

        # --- D. RETURN RESPONSE ---
        return {
            "status": "success",
            "endpoint": "dashboard-analyze",
            "data": {
                "exp_earned": exp_earned,
                "gamification": {
                    "league": user_league,
                    "mission": dynamic_mission, 
                    "budget_utilization_percentage": round(budget_utilization * 100, 1)
                }
                # "ai_advisor_message" DIHAPUS sesuai permintaan Rayza
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# %% 6. ENDPOINT SEKUNDER: Gamification Only
@app.post("/gamification-only")
async def analyze_gamification(request: GamificationOnlyRequest):
    if any(m is None for m in [scaler_km, kmeans_model]):
        raise HTTPException(status_code=500, detail="Model Gamifikasi belum dimuat.")

    try:
        if request.monthly_budget <= 0:
            raise HTTPException(status_code=400, detail="monthly_budget tidak boleh 0 atau minus.")
            
        budget_utilization = request.total_spent / request.monthly_budget
        km_features = pd.DataFrame({
            'total_spent': [request.total_spent],
            'transaction_count': [request.transaction_count],
            'budget_utilization': [budget_utilization]
        })
        
        km_scaled = scaler_km.transform(km_features)
        cluster_id = kmeans_model.predict(km_scaled)[0]
        
        return {
            "status": "success",
            "endpoint": "gamification-only",
            "data": {
                "league": league_mapping.get(cluster_id, "Unranked"),
                "mission": missions_mapping.get(cluster_id, "Catat transaksimu!"),
                "budget_utilization_percentage": round(budget_utilization * 100, 1)
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# %% 7. Health Check
@app.get("/test")
async def test_endpoint():
    return {"message": "API Server Berjalan Normal!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)