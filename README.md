# Finesse: Gamified Budgeting & Financial Health Scorer
> **Backend Repository: Machine Learning & Generative AI**

Repositori ini secara khusus memuat arsitektur backend Machine Learning untuk aplikasi Finesse. Sistem ini bertugas memproses data transaksi pengguna, memberikan metrik evaluasi berupa *Financial Health Score*, dan mensintesis saran keuangan menggunakan Generative AI. 

Proyek ini dikembangkan sebagai Capstone Project dalam program Coding Camp by DBS Foundation.

---

## | Features

### 1. Financial Health Scoring (Deep Learning)
* Memanfaatkan model Deep Learning yang dibangun menggunakan **TensorFlow Functional API**.
* Model dilatih menggunakan *custom training loop* (`tf.GradientTape`) dengan evaluasi metrik *Mean Absolute Error* (MAE) dan optimasi *early stopping*.
* Mengimplementasikan *custom layer* (`FinesseDenseLayer`) untuk pemrosesan operasi *dense* jaringan saraf secara manual.

### 2. AI Financial Advisor (Generative AI)
* Mengintegrasikan API **Google Gemini** (`gemini-2.5-flash`) untuk memberikan evaluasi naratif berbasis teks.
* Mensintesis hasil analitik skor Deep Learning dan data pengeluaran (anggaran bulanan, total pengeluaran, nominal transaksi) menjadi saran keuangan personal.

### 3. RESTful API
* Berjalan di atas framework **FastAPI** untuk menyediakan antarmuka (endpoint) yang asinkron dan efisien.
* Terdokumentasi secara otomatis melalui **Swagger UI** guna memastikan kelancaran integrasi antara model AI dan *frontend* aplikasi.

---

## | Tech Stack

| Kategori | Teknologi |
|---|---|
| **Bahasa Pemrograman** | Python 3.x |
| **Deep Learning Framework** | TensorFlow, Keras |
| **Data Processing & ML** | Scikit-Learn, Pandas, NumPy |
| **Generative AI** | Google GenAI SDK (`google-genai`) |
| **API Framework** | FastAPI, Uvicorn |
| **Utilities** | Joblib, python-dotenv, TensorBoard |

---

## | Project Directory

```text
.
├── dataset/                
├── logs/                   
├── notebooks/              
├── saved_models/           
├── src/                    
├── .env                    
├── .gitignore              
├── main.py                 
├── requirements.txt        
└── README.md
```

## | Panduan Instalasi & Penggunaan Lokal
### 1. Clone Repository
```bash
git clone [URL_REPOSITORY_ANDA]
cd [NAMA_FOLDER_REPOSITORY]
```

### 2. Persiapan Virtual Environment
```bash
python -m venv venv

# Linux/macOS:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Environment Variables
Buat file bernama .env di direktori utama proyek (root directory). Tambahkan API Key Google Gemini Anda ke dalam file tersebut:
```env
GEMINI_API_KEY=masukkan_api_key_gemini_anda_di_sini
```

### 5. Menjalankan Server
Jalankan server Uvicorn untuk memulai FastAPI:
```bash
uvicorn main:app --reload
```

### 6. Pengujian Endpoint
Akses dokumentasi interaktif melalui browser pada alamat http://127.0.0.1:8000/docs untuk melakukan uji coba request JSON langsung ke endpoint /predict.

## | Tentang Kami
Proyek ini dikembangkan secara kolaboratif oleh tim dari Universitas Sumatera Utara:
- Patrick Nicxon Hutabarat (CFCC319D6Y0190) - Full-Stack Web Developer 
- Dame Theresia Rejeki Sidauruk (CDCC319D6X0998) - Data Science 
- Cikita Natasya Br Sembiring (CDCC319D6X1254) - Data Scientist 
- Rayza Indafri Yahya (CACC319D6Y0343) - AI Engineer 
- Samuel Gautama Manik (CACC319D6Y1720) - AI Engineer