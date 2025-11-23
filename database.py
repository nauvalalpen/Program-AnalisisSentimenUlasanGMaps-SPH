import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "sentimen_sph.db"

def init_db():
    """Inisialisasi Database dan Tabel jika belum ada"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT,
            ulasan TEXT,
            prediksi TEXT,
            confidence REAL,
            model_used TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_prediction(ulasan, prediksi, confidence, model_used):
    """Menyimpan hasil prediksi baru"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO history (waktu, ulasan, prediksi, confidence, model_used) VALUES (?, ?, ?, ?, ?)',
              (waktu, ulasan, prediksi, confidence, model_used))
    conn.commit()
    conn.close()

def load_history():
    """Mengambil semua data history"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY id DESC", conn)
    conn.close()
    return df

def clear_history():
    """Menghapus semua history (opsional)"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()