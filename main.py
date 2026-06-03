# Import Libraries
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

# 1. Generative AI Configuration
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("API Key belum diset! Pastikan file .env sudah benar.")

# Initialize GenAI Client
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. Redefine Custom Layer for Model Loading
class FinesseDenseLayer(tf.keras.layers.Layer):
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

# 3. App Initialization & Model Loading
app = FastAPI(
    title="Finesse AI API",
    description="Self-contained API for predicting Financial Health Score using a custom DNN model and providing AI-generated financial advice.",
    version="1.0.0"
)

model = None
preprocessor_dnn = None
target_scaler = None

# Determine Project Root
try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()

try:
    print("\nSearching for and loading DL model and preprocessor...")
    
    model_path = PROJECT_ROOT / 'saved_models' / 'finesse_dnn_v1.keras'
    prep_path = PROJECT_ROOT / 'saved_models' / 'preprocessor_dnn.pkl'
    scaler_path = PROJECT_ROOT / 'saved_models' / 'target_scaler.pkl'
    
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={'FinesseDenseLayer': FinesseDenseLayer}
    )
    preprocessor_dnn = joblib.load(prep_path)
    target_scaler = joblib.load(scaler_path)
    
    print("Deep Learning model and preprocessor loaded successfully!\n")
except Exception as e:
    print(f"\nFailed to load model: {e}")

# 4. Input Schema Validation
class PredictionRequest(BaseModel):
    features: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "amount": 50000,
                    "monthly_budget": 3000000,
                    "cumulative_spend": 1500000,
                    "transaction_to_budget_ratio": 0.016,
                    "budget_utilization_ratio": 0.5,
                    "user_avg_transaction": 45000,
                    "amount_vs_user_avg": 1.1,
                    "is_subscription": False,
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

# 5. Prediction Endpoint & GenAI Integration
@app.post("/predict")
async def predict_financial_health(request: PredictionRequest):
    if preprocessor_dnn is None or model is None or target_scaler is None:
        raise HTTPException(status_code=500, detail="DL model or preprocessor not loaded.")

    try:
        # --- A. Deep Learning Phase (Prediction) ---
        df_input = pd.DataFrame([request.features])
        
        for col in df_input.columns:
            if df_input[col].dtype == 'bool':
                df_input[col] = df_input[col].astype(int)
                
        X_processed = preprocessor_dnn.transform(df_input)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        scaled_prediction = model.predict(X_processed, verbose=0)
        true_prediction = target_scaler.inverse_transform(scaled_prediction)
        final_score = round(float(true_prediction[0][0]), 2)
        
        # --- B. Generative AI Phase (Content Synthesis) ---
        prompt = f"""
        Kamu adalah Finesse, seorang penasihat keuangan pribadi AI yang ramah, ringkas, dan memotivasi. 
        Seorang pengguna baru saja melakukan transaksi dengan detail berikut:
        - Jumlah Transaksi: Rp {request.features.get('amount', 0)}
        - Anggaran Bulanan: Rp {request.features.get('monthly_budget', 0)}
        - Total Pengeluaran Bulan Ini: Rp {request.features.get('cumulative_spend', 0)}
        
        Setelah transaksi ini, AI Deep Learning kami memberikan 'Financial Health Score' sebesar {final_score}/100.
        
        Berdasarkan data tersebut, berikan saran singkat (maksimal 5 kalimat) tentang kesehatan finansial mereka dan apa yang harus dilakukan selanjutnya. Gunakan gaya bahasa kasual.
        """
        
        try:
            # Execution of GenAI content generation
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            ai_advice = response.text
        except Exception as e_genai:
            ai_advice = "Skor berhasil diprediksi, namun AI Advisor sedang sibuk. Tetap bijak dalam mengatur keuanganmu!"
            print(f"Error GenAI: {e_genai}")

        # --- C. Return Results ---
        return {
            "status": "success",
            "financial_health_score": final_score,
            "ai_advisor_message": ai_advice
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occured while processing request: {str(e)}")

@app.get("/test")
async def test_endpoint():
    return {"message": "Test endpoint is working!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)