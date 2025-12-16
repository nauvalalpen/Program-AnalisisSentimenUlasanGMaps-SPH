# ===================================================================
# Tahap 1: Gunakan base image Python yang resmi dan ringan
# ===================================================================
FROM python:3.10-slim

# ===================================================================
# Tahap 2: Set direktori kerja di dalam container
# Ini adalah folder di mana kode kita akan berada
# ===================================================================
WORKDIR /app

# ===================================================================
# Tahap 3: Instal dependensi/library
# Menyalin requirements.txt terlebih dahulu memanfaatkan caching Docker.
# Jika file ini tidak berubah, Docker tidak akan menginstal ulang library
# setiap kali Anda membangun image, sehingga prosesnya lebih cepat.
# ===================================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ===================================================================
# Tahap 4: Salin seluruh kode aplikasi ke dalam container
# Titik (.) pertama berarti "semua file dari direktori saat ini di komputer Anda".
# Titik (.) kedua berarti "salin ke direktori kerja saat ini di dalam container" (/app).
# ===================================================================
COPY . .

# ===================================================================
# Tahap 5: Ekspos port
# Memberitahu Docker bahwa aplikasi di dalam container ini akan 
# mendengarkan di port 5000.
# ===================================================================
EXPOSE 5000

# ===================================================================
# Tahap 6: Perintah untuk menjalankan aplikasi
# Ini adalah perintah yang akan dieksekusi saat container dimulai.
# Kita menggunakan Gunicorn untuk menjalankan aplikasi dari file app.py
# (variabel Flask di dalamnya harus bernama 'app').
# --bind 0.0.0.0:5000: Membuat server dapat diakses dari luar container.
# --workers 4: Menjalankan 4 proses untuk menangani request secara paralel.
# ===================================================================
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]