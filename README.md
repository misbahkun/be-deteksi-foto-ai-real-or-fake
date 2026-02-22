# 🧠 Backend API - Deteksi Gambar Asli vs AI

Repositori ini berisi *source code* untuk Backend API sistem deteksi gambar apakah sebuah foto adalah asli (Real) atau hasil *generate* AI (Artificial). Sistem ini dirancang khusus untuk berjalan secara efisien di lingkungan CPU (tanpa GPU) menggunakan optimasi **ONNX Runtime**, dan akan dikonsumsi oleh aplikasi *mobile* berbasis **Flutter**.

---

## 🏗️ 1. Arsitektur Sistem

Arsitektur aplikasi dibangun dengan pendekatan *Client-Server* menggunakan arsitektur *microservices* ringan.

```text
📱 Client (Flutter App)
       │
       ▼  HTTP POST /predict (multipart/form-data)
┌──────────────────────────────────────────────────┐
│ 🚀 Backend Server (FastAPI - Dockerized)         │
│                                                  │
│  1. Endpoint Routing & Security (app/main.py)    │
│     - Validasi ukuran file (Max 10MB)            │
│     - Validasi MIME type (JPEG, PNG, WEBP)       │
│                                                  │
│  2. Image Preprocessing (app/model.py)           │
│     - Resize 256x256 (Bicubic Interpolation)     │
│     - ImageNet Normalization (Mean & Std)        │
│                                                  │
│  3. AI Inference Engine (ONNX Runtime)           │
│     - Model: SwinV2 (INT8 Quantized)             │
│     - Multi-threading: 4 intra-op threads        │
│                                                  │
└──────────────────────────────────────────────────┘
       │
       ▼  JSON Response
{
  "label": "real",
  "confidence": 0.9999,
  "probabilities": {"artificial": 0.0001, "real": 0.9999},
  "inference_time_ms": 1096
}
```

---

## ⚙️ 2. Flow Proses & Cara Kerja (Pipeline)

Ketika *user* mengunggah gambar dari aplikasi Flutter, berikut adalah alur kerja di sisi *backend*:

1. **Penerimaan Request:** API menerima file gambar. Jika file > 10MB, API langsung menolak request (`HTTP 413 Payload Too Large`) untuk mencegah kehabisan memori server (OOM).
2. **Validasi Tipe Data:** Menggunakan pustaka `python-magic` untuk membaca *header* byte file (bukan sekadar ekstensi) guna memastikan file benar-benar gambar.
3. **Pre-processing Gambar:**
   - Dikonversi ke ruang warna `RGB`.
   - Di-*resize* menjadi resolusi `256 x 256` pixel menggunakan metode *Bicubic*.
   - Dinormalisasi menggunakan standar *ImageNet* (Pixel diubah ke rentang nilai 0-1, lalu dikurangi *mean* dan dibagi *standard deviation*).
   - Diubah bentuk (transpose) dari HWC (Height-Width-Channel) menjadi bentuk matriks CHW `[1, 3, 256, 256]` yang dipahami oleh model.
4. **Inferensi AI:** Matriks gambar dimasukkan ke **ONNX Runtime**. Model melakukan perhitungan matematis dan mengeluarkan *Logits* (nilai probabilitas mentah).
5. **Softmax:** Nilai *Logits* dimasukkan ke fungsi matematika *Softmax* untuk mengubahnya menjadi persentase probabilitas (0.0 hingga 1.0) dengan total jumlah pasti 1 (atau 100%).

---

## 🤖 3. Model AI & Dataset (Penting untuk Sidang)

### Spesifikasi Model: `Modotte/AIRealNet`
- **Arsitektur Dasar:** SwinV2 Transformer (Swinv2ForImageClassification).
- **Target Deteksi:** Gambar asli kamera vs Gambar buatan AI Modern (Midjourney, DALL-E, Stable Diffusion).
- **Format:** Di-eksport dari PyTorch (`.safetensors`) ke **ONNX**.

### 💡 Keputusan Engineering (Engineering Decisions)

1. **Kenapa ONNX Runtime, bukan PyTorch?**
   PyTorch memiliki ukuran *library* yang sangat besar (~2.5GB) dan memakan banyak RAM. ONNX Runtime dirancang khusus untuk tahap produksi (*inference*) sehingga jauh lebih cepat, ringan, dan ukuran Docker image menyusut drastis.
2. **Kenapa menggunakan Quantization INT8 (QUInt8)?**
   Server *homelab* (Proxmox dengan Intel i5-3470) adalah prosesor generasi lama yang **tidak memiliki instruksi AVX2**. Jika menggunakan model bawaan (FP32), proses akan sangat lambat atau bahkan *crash*. Dengan teknik *Dynamic Quantization* (QUInt8), bobot model dikompresi dari **~744 MB** menjadi **~190 MB**. Ini membuat model bisa berjalan sangat cepat (~1.2 detik per gambar) hanya dengan bermodalkan CPU tua dan RAM 6GB.

### Catatan Tentang Dataset `CIFAKE`
Jika dalam pengujian menggunakan dataset **CIFAKE** model terkadang keliru, hal tersebut memiliki penjelasan teknis:
- **CIFAKE** berisi gambar hasil *generate* dari teknologi lawas (GAN) dengan resolusi sangat kecil (awalnya 32x32 piksel dari CIFAR-10) yang diperbesar.
- Model **AIRealNet (SwinV2)** dilatih secara khusus untuk mendeteksi *pattern* atau ciri khas dari *Generative AI* modern beresolusi tinggi (seperti tekstur kulit Midjourney atau artefak jari Stable Diffusion). Gambar CIFAKE yang *pixelated* (pecah/buram) seringkali dianggap model sebagai foto asli beresolusi rendah yang diambil dengan kamera jelek, bukan buatan AI modern.

---

## 💻 4. Tech Stack

- **Framework API:** `FastAPI` (Cepat, berbasis *Async*, Auto-generates Swagger Docs).
- **Server Gateway:** `Uvicorn`
- **Machine Learning Engine:** `onnxruntime`
- **Image Processing:** `Pillow` (PIL) & `numpy`
- **Deployment:** `Docker` & `Docker Compose`

---

## 📡 5. Dokumentasi API (Endpoints)

### `GET /health`
Mengecek apakah server hidup dan model berhasil dimuat ke dalam RAM.
- **Response `200 OK`**:
  ```json
  {"status": "ok", "model_loaded": true}
  ```

### `POST /predict`
Mendeteksi gambar. Parameter dikirim sebagai `multipart/form-data` dengan key `file`.
- **Response `200 OK`**:
  ```json
  {
    "label": "real",
    "confidence": 0.9999970197677612,
    "probabilities": {
      "artificial": 2.9237223770905985e-06, 
      "real": 0.9999970197677612
    },
    "inference_time_ms": 1096
  }
  ```
  *(Catatan: `e-06` dalam notasi saintifik berarti `0.0000029`)*.

---

## 🚀 6. Cara Menjalankan (Deployment)

Proyek ini telah dibungkus ke dalam *Docker*, sehingga sangat mudah dijalankan di Virtual Machine (Proxmox/Ubuntu) maupun VPS cloud tanpa perlu menginstal Python secara manual.

### Prasyarat:
- Pastikan file model ONNX sudah ada di dalam folder `models/model.onnx`
- Docker & Docker Compose sudah terinstal di server.

### Perintah Menjalankan:
```bash
# 1. Masuk ke direktori proyek
cd be-deteksi-foto-ai-real-or-fake

# 2. Jalankan container di background (detached mode)
docker compose up -d

# 3. Melihat log server secara realtime (opsional)
docker compose logs -f
```

Server API akan langsung berjalan di port `8000` dan siap diakses oleh aplikasi Flutter. 
Buka `http://<IP-SERVER>:8000/docs` di browser untuk mencoba API secara interaktif melalui Swagger UI bawaan FastAPI.
