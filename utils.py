import requests
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile

def load_lottieurl(url: str):
    """Fungsi untuk memuat animasi Lottie dari URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

class PDFReport(FPDF):
    def header(self):
        # Logo bisa ditambahkan jika ada file gambar lokal
        # self.image('logo.png', 10, 8, 33)
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Laporan Analisis Sentimen - Semen Padang Hospital', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(total_ulasan, pos_pct, neg_pct, acc_score, f1_score, fig_pie, fig_cm):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Bagian 1: Ringkasan Eksekutif
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "1. Ringkasan Eksekutif", 0, 1)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"Berdasarkan analisis terhadap {total_ulasan} data ulasan Google Maps, "
                          f"ditemukan bahwa {pos_pct:.1f}% ulasan bernada POSITIF dan {neg_pct:.1f}% bernada NEGATIF. "
                          f"Model AI memiliki tingkat akurasi sebesar {acc_score:.1f}%.")
    pdf.ln(5)
    
    # Bagian 2: Statistik Visual
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "2. Visualisasi Data", 0, 1)
    
    # Simpan plot sementara ke file gambar agar bisa masuk PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile_pie:
        fig_pie.write_image(tmpfile_pie.name) # Plotly save
        pdf.image(tmpfile_pie.name, x=10, y=None, w=100)
    
    # Bagian 3: Rekomendasi
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "3. Rekomendasi Manajemen", 0, 1)
    pdf.set_font("Arial", size=12)
    
    if neg_pct > 20:
        rekomendasi = "- Perlu evaluasi mendalam pada aspek pelayanan antrian.\n- Tingkatkan responsivitas staf di jam sibuk."
    else:
        rekomendasi = "- Pertahankan kualitas layanan saat ini.\n- Fokus pada mempertahankan kepuasan pelanggan."
        
    pdf.multi_cell(0, 10, rekomendasi)
    
    return pdf.output(dest='S').encode('latin-1')