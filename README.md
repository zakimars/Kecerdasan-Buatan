# 🎓 Tugas Akhir - Kecerdasan Buatan

Repositori ini berisi kumpulan tugas akhir saya untuk mata kuliah **Kecerdasan Buatan**. Seluruh proyek dikerjakan secara mandiri dengan memanfaatkan model **CNN**, **OpenCV**, dan berbagai dataset publik dari **Kaggle**.

---

## 📁 Daftar File

### 1. `Tugas_Akhir_AI.ipynb` — Deteksi Emosi Wajah & Rekomendasi Film

> **Dataset:** [FER2013 - Facial Expression Recognition](https://www.kaggle.com/datasets/msambare/fer2013)

Terdiri dari dua bagian utama:

- **🧠 Deteksi Emosi Wajah**  
  Menggunakan CNN untuk mengenali ekspresi wajah dari gambar grayscale. Model mengenali 7 jenis emosi:  
  `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, dan `Neutral`.

- **🎬 Rekomendasi Film Berdasarkan Emosi (OCR + Mistral AI)**  
  - Menggunakan `easyOCR` untuk membaca emosi dari gambar hasil prediksi (misalnya: "happy", "sad").  
  - Emosi digunakan sebagai input untuk prompt yang dikirim ke **Mistral AI API**.  
  - Model LLM Mistral merespons dengan 3 rekomendasi film dalam bahasa Indonesia.  
  - Disajikan melalui antarmuka interaktif menggunakan **Gradio**.

---

### 2. `Tugas_PR_AI.ipynb` — Klasifikasi Gambar Bunga

> **Dataset:** [Flowers Recognition (Kaggle)](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)

- Mendeteksi 5 jenis bunga:  
  `Daisy`, `Dandelion`, `Rose`, `Sunflower`, `Tulip`.  
- Menggunakan arsitektur CNN untuk klasifikasi gambar.

---

### 3. `opencvAI.py` — Deteksi Emosi Wajah Real-Time

- Aplikasi real-time menggunakan **OpenCV**.
- Menggunakan model `.keras` yang dihasilkan dari `Tugas_Akhir_AI.ipynb`.
- Melakukan deteksi ekspresi wajah secara langsung melalui webcam.

---

## ⚠️ Catatan

- Seluruh kode dikembangkan untuk keperluan akademik.
- File model (.keras) dan dataset tidak disertakan dalam repositori ini karena ukurannya besar.  
  Silakan unduh secara manual melalui link Kaggle yang telah disediakan.

---

## 👨‍💻 Author

**Zaki Marsyandi**  
Semester 6 – Teknik Elektro  
Mata Kuliah: Kecerdasan Buatan
