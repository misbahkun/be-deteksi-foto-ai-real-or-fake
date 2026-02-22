# 🧠 Backend API - PindAI (Deteksi Gambar Asli vs AI)

Repositori ini berisi *source code* untuk Backend API **PindAI**, sebuah sistem cerdas yang memindai apakah sebuah foto adalah asli (Real) tangkapan kamera atau hasil *generate* AI (Artificial) seperti Midjourney, DALL-E, dan Stable Diffusion. 

Sistem ini dibangun untuk mendemonstrasikan implementasi *Computer Vision* pada lingkungan dengan sumber daya terbatas (Resource-Constrained Environments).Menggunakan arsitektur *microservices* yang sangat ringan, API ini berjalan secara efisien di lingkungan server CPU tua (tanpa GPU) berkat optimasi kompresi **ONNX Runtime (INT8)**, dan dirancang untuk dikonsumsi langsung oleh aplikasi *mobile* berbasis **Flutter**.

---

## 📁 1. Struktur Direktori Proyek

```text
be-deteksi-foto-ai-real-or-fake/
├── app/                      # Inti dari Backend API
│   ├── main.py             # Routing FastAPI, Konfigurasi CORS, & Endpoint
│   ├── model.py            # Logika Preprocessing & Inferensi ONNX Runtime
│   └── schemas.py          # Pydantic Model (Format response JSON)
├── models/                   # Tempat penyimpanan file model AI
│   └── model.onnx          # Model SwinV2 INT8 Quantized (~190MB)
├── scripts/                  # Skrip utilitas/bantuan
│   └── export_onnx.py      # Skrip untuk download & convert model dari HuggingFace ke ONNX
├── Dockerfile                # Blueprint untuk merakit kontainer server aplikasi
├── requirements.txt          # Library Python khusus untuk Production (Tanpa PyTorch)
├── requirements-dev.txt      # Library Python untuk Development (Transformers, Optimum, dll)
└── README.md                 # Dokumentasi Proyek ini
```

---

## 🏗️ 2. Arsitektur Sistem

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
│     - Resize 256x256 (Bilinear Interpolation)    │
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

## ⚙️ 3. Flow Proses & Cara Kerja (Pipeline)

Ketika *user* mengunggah gambar dari aplikasi Flutter, berikut adalah alur kerja di sisi *backend*:

1. **Penerimaan Request:** API menerima file gambar via jaringan. Jika file > 10MB, API langsung menolak request (`HTTP 413 Payload Too Large`) untuk mencegah kehabisan memori server (OOM).
2. **Validasi Tipe Data:** Menggunakan pustaka `python-magic` untuk membaca *header* byte file (bukan sekadar ekstensi palsu) guna memastikan file benar-benar gambar.
3. **Pre-processing Gambar:**
   - Dikonversi ke ruang warna `RGB`.
   - Di-*resize* menjadi resolusi `256 x 256` pixel menggunakan metode *Bilinear* (mengikuti konfigurasi `AutoImageProcessor` dari pencipta model asli).
   - Dinormalisasi menggunakan standar *ImageNet* (Pixel diubah ke rentang nilai 0-1, lalu dikurangi *mean* dan dibagi *standard deviation*).
   - Diubah bentuk (transpose) dari HWC (Height-Width-Channel) menjadi bentuk matriks CHW `[1, 3, 256, 256]` yang dipahami oleh model.
4. **Inferensi AI:** Matriks gambar dimasukkan ke **ONNX Runtime**. Model mengekstraksi dan mempelajari pola *noise* atau artefak di dalam gambar untuk mengeluarkan *Logits* (nilai probabilitas mentah).
5. **Softmax:** Nilai *Logits* dimasukkan ke fungsi matematika *Softmax* untuk mengubahnya menjadi persentase probabilitas (0.0 hingga 1.0) dengan total jumlah pasti 1 (atau 100%).
6. **Response JSON:** Backend mengirimkan hasil (`label` terbesar beserta `confidence`) kembali ke layar HP pengguna (*Flutter App*).

---

## 🤖 4. Model AI & Dataset

### Spesifikasi Model: `Modotte/AIRealNet`
- **Creator/Publisher:** Modotte (Tersedia publik di [HuggingFace - Modotte/AIRealNet](https://huggingface.co/Modotte/AIRealNet))
- **Arsitektur Dasar:** SwinV2 Transformer (Swinv2ForImageClassification - Tiny Version). SwinV2 adalah *Vision Transformer* buatan Microsoft yang sangat canggih dalam mengenali pola hierarkis pada gambar.
- **Dataset Pelatihan Utama (Fine-Tuning Data):** 
  Berbeda dengan model raksasa yang sering menggunakan dataset kontroversial (seperti LAION yang melakukan *scraping* web tanpa izin/*consent*), model [`Modotte/AIRealNet`](https://huggingface.co/Modotte/AIRealNet) di-*fine-tune* secara khusus menggunakan dataset yang lebih kecil, terkurasi (*curated*), dan sadar privasi (*privacy-conscious*), yaitu **[`Parveshiiii/AI-vs-Real`](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real)** (tersedia publik di HuggingFace).
  1. Dataset ini dirancang spesifik hanya untuk *task detection* (klasifikasi), bukan untuk *generative pretraining*, sehingga fokusnya sangat tajam pada membedakan fitur visual asli dan buatan.
  2. Pendekatan ini memastikan bahwa aplikasi **PindAI** menjunjung tinggi Etika AI (*AI Ethics*) dengan menghindari penggunaan dataset bias atau yang melanggar hak cipta massal.
- **Sumber Jurnal/Referensi Pendukung:** Pendekatan menggunakan *Vision Transformer* (SwinV2) merujuk pada berbagai penelitian deteksi AI modern yang membuktikan bahwa arsitektur berbasis *Attention Mechanism* (ViT/Swin) lebih superior dibanding CNN tradisional (ResNet) dalam menangkap pola artefak mikro (*micro-artifacts*) buatan AI.
- **Cara Kerja Deteksi (Fitur Utama):** Model tidak hanya melihat "apakah gambar ini bagus atau tidak", melainkan menganalisis **artefak mikro tingkat piksel (micro-artifacts)** yang sering ditinggalkan oleh *Generative AI*, seperti pola tekstur kulit yang tidak natural, asimetri pada pupil mata, pola asimetris pada latar belakang (*background*), hingga struktur geometri rambut yang tidak logis.
- **Format:** Di-eksport dari lingkungan PyTorch (`.safetensors`) ke format **ONNX** untuk optimasi server.

### 💡 Keputusan Engineering (Engineering Decisions)

1. **Kenapa menggunakan FastAPI dibandingkan Flask (yang dipakai pembuat model asli)?**
   - **Performa & Asynchronous:** FastAPI dibangun menggunakan arsitektur *asynchronous* (ASGI), sedangkan Flask secara *default* adalah *synchronous* (WSGI). Untuk model AI yang memakan waktu kalkulasi (misal ~1 detik), menggunakan `async`/`await` pada FastAPI memastikan server tidak "membeku" (*blocking*) saat menerima request gambar lain secara bersamaan.
   - **Validasi Data Otomatis:** FastAPI menggunakan *Pydantic* untuk memvalidasi tipe data JSON *Response*. Jika ada kesalahan tipe data (misal: confidence tiba-tiba berubah jadi *string*), API akan otomatis menolaknya dan mengembalikan error yang terstruktur, menjaga agar aplikasi Flutter kamu tidak *crash* karena *parsing error*.
   - **Dokumentasi Otomatis:** FastAPI secara otomatis membuat halaman dokumentasi interaktif (Swagger UI) di endpoint `/docs`. Fitur ini sangat mempermudah *Frontend Developer* (tim Flutter) dalam melakukan integrasi API tanpa memerlukan dokumen eksternal seperti Postman.

2. **Apa itu ONNX Runtime dan Kenapa menggunakannya (bukan PyTorch)?**
   - **ONNX (Open Neural Network Exchange)** adalah format standar terbuka untuk merepresentasikan model *machine learning*. **ONNX Runtime** adalah mesin/engine berkecepatan tinggi buatan Microsoft untuk menjalankan model ONNX.
   - PyTorch sangat bagus untuk melatih model (*Training*), tetapi sangat rakus memori dan lambat saat di-*deploy* ke *server* untuk digunakan (fase *Inference*). Pustaka PyTorch bisa memakan penyimpanan hingga ~2.5GB.
   - Dengan mengonversi model ke format ONNX dan menggunakan `onnxruntime` saja, ukuran dependensi proyek menyusut drastis, penggunaan RAM jauh lebih kecil, dan kecepatan pemrosesan (*inference speed*) meningkat secara signifikan karena ONNX melakukan optimasi pada tingkat struktur *graph* matematis model.

3. **Kenapa menggunakan Quantization INT8 (QUInt8)?**
   Server *homelab* (Proxmox dengan Intel i5-3470) adalah prosesor generasi lama yang **tidak memiliki instruksi AVX2** (Advanced Vector Extensions). Jika menggunakan model bawaan (FP32/Float32), proses perhitungan matriks akan sangat lambat atau bahkan server menjadi *crash* (*Illegal Instruction*). Dengan teknik *Dynamic Quantization* (QUInt8), bobot model dikompresi (diturunkan presisinya) dari **~744 MB** menjadi **~190 MB**. Ini membuat model bisa berjalan sangat cepat (~1.1 detik per gambar) hanya dengan bermodalkan CPU tua.

4. **Kenapa gambar harus di-*resize* ke 256x256 pixel?**
   - Model SwinV2 (dan mayoritas *Vision Transformer*) dilatih dengan struktur *input* berukuran tetap (Fixed Tensor Shape). Arsitektur `AIRealNet` secara bawaan memotong gambar menjadi *patch-patch* kecil berukuran tertentu. Jika gambar yang dikirim dari HP beresolusi 4K (3840x2160) dimasukkan langsung, model tidak akan bisa menghitungnya karena bentuk matriks matematisnya tidak cocok. Karena itu, gambar apa pun harus dipaksa menjadi kotak 256x256 piksel sebelum masuk ke model.

5. **Kenapa menggunakan *Bilinear Interpolation* (`Image.Resampling.BILINEAR`) saat *resize*?**
   - Saat sebuah gambar diperkecil/diperbesar (*rescaling*), piksel baru harus diciptakan/dibuang. Terdapat berbagai metode matematis untuk ini: *Nearest Neighbor* (kasar/berkotak), *Bilinear* (rata-rata linier antar piksel), atau *Bicubic* (lebih halus dan tajam).
   - Awalnya kita menggunakan *Bicubic*, tetapi setelah membedah kode *source* (konfigurasi `AutoImageProcessor`) dari pencipta asli model `Modotte`, ditemukan bahwa model tersebut dilatih (di-*training*) menggunakan kompresi **Bilinear**.
   - Dalam dunia *Computer Vision*, metode *preprocessing* (*resizing*) di tahap *inference* (penggunaan) **harus 100% identik** dengan metode saat *training*. Perbedaan kecil (seperti tepian piksel yang lebih tajam akibat *Bicubic*) dapat membingungkan model (karena dia mencari *micro-artifacts* atau cacat kecil buatan AI) dan menurunkan akurasi deteksinya.

- Model **AIRealNet (SwinV2)** dilatih secara khusus untuk mendeteksi *pattern* atau ciri khas dari *Generative AI* modern beresolusi tinggi (seperti tekstur kulit Midjourney atau artefak jari Stable Diffusion). Gambar CIFAKE yang *pixelated* (pecah/buram) seringkali dianggap model sebagai foto asli beresolusi rendah yang diambil dengan kamera jelek, bukan buatan AI modern.

---

## 💻 6. Tech Stack & Library (Penjelasan Detail)

Berikut adalah daftar pustaka (*library*) utama yang digunakan beserta fungsinya, dibagi berdasarkan lingkungan *Production* dan *Development*:

### A. Lingkungan Server / Production (`requirements.txt`)
Pustaka inti yang wajib diinstal agar API bisa berjalan di server Docker/VPS:
- **`fastapi`**: Framework web modern berbahasa Python. Sangat cepat, mendukung pemrograman *Asynchronous* (`async`/`await`), dan otomatis membuat dokumentasi API (Swagger UI).
- **`uvicorn[standard]`**: ASGI (*Asynchronous Server Gateway Interface*) web server. FastAPI hanyalah kerangka kerja (*framework*), `uvicorn` adalah "mesin" server sungguhan yang mendengarkan request HTTP di port 8000 dan meneruskannya ke FastAPI.
- **`python-multipart`**: Pustaka tambahan wajib untuk FastAPI agar bisa mengurai (parsing) request HTTP berjenis `multipart/form-data` (format standar saat mengirim file gambar dari Flutter ke API). Tanpa ini, FastAPI akan menolak upload file.
- **`python-magic`**: Pustaka keamanan (*security*) untuk membaca "Magic Bytes" (header mentah) dari sebuah file. Digunakan untuk memverifikasi apakah file yang diunggah benar-benar sebuah gambar (JPEG/PNG/WEBP), bukan file berekstensi palsu (misal: `virus.exe` diubah namanya menjadi `gambar.jpg`).
- **`Pillow` (PIL)**: Pustaka manipulasi gambar standar Python. Digunakan untuk memuat (*load*) byte gambar ke dalam memori, mengubahnya ke format RGB, dan melakukan *resize* dengan algoritma `Bilinear`.
- **`numpy`**: Pustaka komputasi matematika tingkat tinggi. Digunakan untuk mengubah gambar dari Pillow menjadi matriks angka (*array*), lalu melakukan normalisasi matematika (membagi nilai piksel dengan 255, mengurangi *Mean*, dan membagi dengan *Standard Deviation*) agar matriks siap dibaca AI.
- **`onnxruntime`**: Mesin inferensi buatan Microsoft. Bertugas mengeksekusi file model `.onnx` dengan matriks gambar dari `numpy` secara sangat efisien di CPU tanpa perlu menginstal framework raksasa seperti PyTorch.

### B. Lingkungan Development (`requirements-dev.txt`)
Pustaka tambahan yang HANYA digunakan oleh programmer di laptop saat mengembangkan aplikasi:
- **`transformers` & `optimum[onnxruntime]`**: Pustaka buatan HuggingFace. Digunakan HANYA di dalam `scripts/export_onnx.py` untuk mengunduh model asli AIRealNet (PyTorch) dari internet, lalu mengonversinya (Export & Quantize) menjadi format `.onnx` yang ringan.

---

## 📡 7. Dokumentasi API (Endpoints)

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

## 🛠️ 8. Panduan Instalasi (Development & Local)

Bagi pengembang (developer) atau penguji yang ingin menjalankan dan memodifikasi proyek ini di komputer lokal (tanpa Docker), ikuti langkah-langkah berikut:

### Prasyarat:
- Python versi **3.10** hingga **3.12** terinstal di komputer.
- OS: Linux (Ubuntu/Debian direkomendasikan), macOS, atau Windows (via WSL2).

### Langkah-langkah Instalasi Lokal:

1. **Clone Repositori:**
   ```bash
   git clone <URL_REPO_KAMU>
   cd be-deteksi-foto-ai-real-or-fake
   ```

2. **Buat Virtual Environment (Sangat Direkomendasikan):**
   Ini untuk memastikan pustaka Python tidak bentrok dengan *project* lain di laptopmu.
   ```bash
   python3 -m venv .venv
   
   # Aktifkan virtual environment (Linux/macOS)
   source .venv/bin/activate
   
   # Atau jika di Windows (Command Prompt / PowerShell)
   .venv\Scripts\activate
   ```

3. **Instal Library Development & Production:**
   Instal file `requirements-dev.txt` yang sudah mencakup pustaka *Production* dan pustaka untuk konversi model.
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Unduh & Konversi Model AI (ONNX Export):**
   *Project* ini tidak menyertakan file model berukuran besar (~190MB) di GitHub. Kamu harus mengunduhnya langsung dari server HuggingFace lalu mengonversinya (Quantization).
   ```bash
   python scripts/export_onnx.py
   ```
   *(Tunggu hingga proses selesai dan file `models/model.onnx` berhasil dibuat. Ukurannya sekitar 190MB).*

5. **Jalankan Server Lokal (Live Reload):**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Buka `http://localhost:8000/docs` di browsermu untuk mulai mengetes API via Swagger UI.

---

## 🚀 9. Cara Menjalankan (Production / Deployment menggunakan Docker)

Proyek ini telah dibungkus dengan Docker. Karena ini adalah *single-container application*, kita bisa langsung menggunakan perintah `docker run` tanpa memerlukan `docker-compose`.

### A. Langkah Build & Push Image (Di Laptop/Mesin Dev)

1. **Build Image Docker:**
   Beri nama image kamu (misalnya menggunakan username Docker Hub kamu `mizzcode`).
   ```bash
   docker build -t mizzcode/ai-image-detector:latest .
   ```

2. **Push ke Docker Hub (Opsional):**
   Jika kamu ingin memindahkan image ke server/homelab dengan mudah.
   ```bash
   docker login
   docker push mizzcode/ai-image-detector:latest
   ```

### B. Langkah Menjalankan di Server / Homelab Proxmox

Karena file model AI (`models/model.onnx` ukuran ~190MB) **tidak dimasukkan (*baked*) ke dalam image Docker** agar image tetap ringan, kamu harus mengunggah file `model.onnx` tersebut secara terpisah ke server homelab kamu.

1. **Siapkan Folder & File Model di Server:**
   ```bash
   # Di dalam server Proxmox/Ubuntu
   mkdir -p ~/ai-detector/models
   ```
   *(Pindahkan/copy file `model.onnx` dari laptop ke folder `~/ai-detector/models` di server menggunakan FileZilla atau SCP).*

2. **Tarik (Pull) Image dari Docker Hub:**
   ```bash
   docker pull mizzcode/ai-image-detector:latest
   ```

3. **Jalankan Container (Docker Run):**
   Gunakan perintah berikut untuk menjalankan server. Perhatikan parameter `-v` (Volume Mount) yang menghubungkan folder `models` di server fisik ke dalam container.
   ```bash
   docker run -d \
     --name ai-detector-api \
     --restart unless-stopped \
     -p 8000:8000 \
     -v $(pwd)/models:/app/models:ro \
     --memory="2g" \
     mizzcode/ai-image-detector:latest
   ```

   **Penjelasan Parameter:**
   - `-d`: Berjalan di latar belakang (*background*).
   - `-p 8000:8000`: Membuka port 8000 agar bisa diakses.
   - `-v ...`: Memasang (*mount*) folder model secara *Read-Only* (`:ro`) agar model tidak termodifikasi.
   - `--memory="2g"`: Membatasi penggunaan RAM maksimal 2GB agar server homelab tidak *crash* (OOM).

4. **Cek Status & Log:**
   ```bash
   docker ps
   docker logs -f ai-detector-api
   ```

Server API akan langsung berjalan di port `8000` dan siap diakses oleh aplikasi Flutter. 
Buka `http://<IP-SERVER>:8000/docs` di browser untuk mencoba API secara interaktif melalui Swagger UI bawaan FastAPI.

---

## 🔄 10. Cara Update Model (Untuk Mendeteksi AI Generasi Baru)

Teknologi *Generative AI* berkembang sangat cepat. Suatu saat, model `Modotte/AIRealNet` mungkin akan kesulitan mendeteksi gambar AI terbaru (misal: Midjourney v7 atau Sora). Sistem ini dirancang sangat dinamis sehingga kamu bisa **mengganti modelnya kapan saja tanpa perlu merombak kode server**.

**Langkah-langkah mengganti model:**
1. Cari model *Image Classification* berbasis **SwinV2** atau **ViT** terbaru di [HuggingFace](https://huggingface.co/models?pipeline_tag=image-classification).
2. Buka file `scripts/export_onnx.py`.
3. Ubah variabel `MODEL_ID` menjadi ID model yang baru (misal: `MODEL_ID = "username/ModelBaruTerhebat"`).
4. Pastikan lingkungan development berjalan (`source .venv/bin/activate`).
5. Jalankan ulang skrip konversi:
   ```bash
   python scripts/export_onnx.py
   ```
6. Jika model baru memiliki nama *Class/Label* yang berbeda (misal: `["fake", "real"]` bukan `["artificial", "real"]`), sesuaikan array variabel `LABELS` di file `app/model.py`.
7. *Restart* container Docker (`docker restart ai-detector-api`). Server kini secara instan menggunakan model AI yang paling baru!

---

## 📈 11. Rekomendasi Pengembangan Lanjutan (Future Work)

Saran Untuk pengembangan di masa depan:

1. **Sistem Autentikasi API (API Key)**
   Saat ini endpoint `/predict` bersifat terbuka (*public*). Ke depannya perlu ditambahkan validasi *Header API Key* atau *token JWT* agar hanya aplikasi Flutter kamu yang bisa menggunakan API ini, guna mencegah *abuse* atau *spam request* dari pihak luar.
2. **Rate Limiting**
   Bisa ditambahkan *rate limiter* (misal: maks 10 request/menit per *user*) untuk mencegah server kewalahan (*overload*/DDoS) dari antrean pemrosesan gambar yang berat.
3. **Dukungan GPU (onnxruntime-gpu)**
   Saat ini sistem berjalan 100% menggunakan CPU. Jika di masa depan server *homelab* di-*upgrade* dengan menambahkan kartu grafis (NVIDIA GPU), library `onnxruntime` di `requirements.txt` dapat diganti menjadi `onnxruntime-gpu` agar kecepatan deteksi melonjak drastis (dari ~1.1 detik menjadi ~0.05 detik per gambar).
4. **Fine-Tuning Model Lokal**
   Akurasi model *pre-trained* sangat bergantung pada dataset global. Untuk hasil yang lebih presisi, model dapat di-*fine-tune* ulang (*transfer learning*) menggunakan kumpulan dataset gambar wajah orang Indonesia yang nyata dikombinasikan dengan gambar buatan AI bertema Indonesia.
