from fpdf import FPDF
import tempfile
import os
import base64
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        # Logo Text / Brand
        self.set_font('Arial', 'B', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Hospital-Check System', 0, 1, 'R')
        self.line(10, 20, 200, 20)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(240, 248, 255) # Alice Blue
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, f"  {title}", 0, 1, 'L', 1)
        self.ln(5)

    def section_subtitle(self, title):
        self.set_font('Arial', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, 0, 1, 'L')

    def add_chart(self, b64_str, h=80, w=140, x=None, y=None):
        """
        Menambahkan gambar dari base64 string.
        PERBAIKAN: Otomatis memindahkan kursor ke bawah setelah gambar.
        """
        if not b64_str: 
            self.ln(5)
            return
            
        try:
            img_data = base64.b64decode(b64_str)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_data)
                tmp_path = tmp_file.name
            
            # Logika Posisi
            if x is None: x = (210 - w) / 2 # Center horizontally
            if y is None: y = self.get_y()  # Current vertical position
            
            self.image(tmp_path, x=x, y=y, w=w, h=h)
            os.unlink(tmp_path)
            
            # --- PERBAIKAN UTAMA DI SINI ---
            # Jika posisi default (bukan custom x/y), paksa kursor turun
            # Turun sebesar Tinggi Gambar (h) + Margin (5)
            if x == (210 - w) / 2: 
                self.set_y(y + h + 5)
            # -------------------------------
                
        except Exception as e:
            print(f"PDF Image Error: {e}")
            self.cell(0, 10, "[Gambar Tidak Tersedia]", 0, 1)

def clean_text(text):
    if not isinstance(text, str): return str(text)
    return text.encode('latin-1', 'ignore').decode('latin-1')

# --- FUNGSI 1: LAPORAN GLOBAL ---
def create_global_report(metrics, stats, learning_curve, wc_pos, wc_neg, hospital_ranks, pie_chart):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Halaman 1: Ringkasan
    pdf.add_page()
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, "Laporan Analisis Sentimen (Global)", 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Tanggal: {datetime.now().strftime('%d %B %Y')}", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.chapter_title("1. Ringkasan Eksekutif")
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 10, f"Total Laporan Masuk: {stats['total']}", 0, 1)
    
    # Pie Chart
    if pie_chart:
        pdf.add_chart(pie_chart, h=90)
    else:
        pdf.ln(10)

    # Halaman 2: Evaluasi Model
    pdf.add_page()
    pdf.chapter_title("2. Evaluasi Model AI")
    pdf.cell(0, 10, f"Akurasi Model: {metrics.get('accuracy', 0)}%", 0, 1)
    
    # Learning Curve
    if learning_curve:
        pdf.add_chart(learning_curve, h=80)
    else:
        pdf.ln(10)

    # Halaman 3: Wordcloud Global
    pdf.add_page()
    pdf.chapter_title("3. Peta Kata Kunci Global")
    
    # Wordcloud Negatif
    pdf.section_subtitle("Topik Keluhan (Sentimen Negatif):")
    if wc_neg:
        pdf.add_chart(wc_neg, h=75) # Fungsi add_chart yang baru akan otomatis menurunkan kursor
    else:
        pdf.cell(0, 10, "[Belum cukup data negatif]", 0, 1, 'C')
        
    # Wordcloud Positif (Sekarang aman, tidak akan menumpuk)
    pdf.section_subtitle("Topik Apresiasi (Sentimen Positif):")
    if wc_pos:
        pdf.add_chart(wc_pos, h=75)
    else:
        pdf.cell(0, 10, "[Belum cukup data positif]", 0, 1, 'C')

    # Halaman 4: Peringkat RS
    pdf.add_page()
    pdf.chapter_title("4. Peringkat Performa Rumah Sakit")
    
    # Header Tabel
    pdf.set_font('Arial', 'B', 9)
    pdf.set_fill_color(50, 50, 50)
    pdf.set_text_color(255, 255, 255)
    
    pdf.cell(10, 10, "No", 1, 0, 'C', 1)
    pdf.cell(70, 10, "Nama Rumah Sakit", 1, 0, 'C', 1)
    pdf.cell(20, 10, "Total", 1, 0, 'C', 1)
    pdf.cell(20, 10, "Positif", 1, 0, 'C', 1)
    pdf.cell(20, 10, "Negatif", 1, 0, 'C', 1)
    pdf.cell(30, 10, "Kepuasan", 1, 1, 'C', 1)
    
    # Isi Data
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(0, 0, 0)
    
    for i, rs in enumerate(hospital_ranks, 1):
        name = clean_text(rs['name'])
        if len(name) > 30: name = name[:27] + "..."
        neg_count = rs['total'] - rs['pos']
        
        fill = True if i % 2 == 0 else False
        pdf.set_fill_color(245, 245, 245)

        pdf.cell(10, 8, str(i), 1, 0, 'C', fill)
        pdf.cell(70, 8, f" {name}", 1, 0, 'L', fill)
        pdf.cell(20, 8, str(rs['total']), 1, 0, 'C', fill)
        
        pdf.set_text_color(0, 100, 0)
        pdf.cell(20, 8, str(rs['pos']), 1, 0, 'C', fill)
        
        pdf.set_text_color(180, 0, 0)
        pdf.cell(20, 8, str(neg_count), 1, 0, 'C', fill)
        
        pdf.set_text_color(0, 0, 0)
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(30, 8, f"{rs['score']}%", 1, 1, 'C', fill)
        pdf.set_font('Arial', '', 9)

    return pdf.output(dest='S')

# --- FUNGSI 2: LAPORAN SPESIFIK ---
def create_specific_report(rs_name, stats_past, stats_live, analysis_neg, analysis_pos, wc_neg, wc_pos, samples):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    clean_rs_name = clean_text(rs_name)
    
    # Header
    pdf.set_font('Arial', 'B', 18)
    pdf.cell(0, 10, f"Laporan Audit Kualitas Layanan", 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"{clean_rs_name}", 0, 1, 'C')
    pdf.line(10, 30, 200, 30)
    pdf.ln(5)

    # 1. Scorecard
    pdf.chapter_title("1. Ringkasan Kinerja (Scorecard)")
    
    raw_trend = stats_live['score'] - stats_past['score']
    trend = round(raw_trend, 1)
    trend_symbol = "+" if trend >= 0 else ""
    
    pdf.set_font('Arial', '', 10)
    pdf.cell(95, 8, "Skor Historis (Masa Lalu)", 1, 0, 'C')
    pdf.cell(95, 8, "Skor Saat Ini (Real-time)", 1, 1, 'C')
    
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(95, 15, f"{stats_past['score']}%", 1, 0, 'C')
    
    if trend >= 0: pdf.set_text_color(0, 100, 0)
    else: pdf.set_text_color(200, 0, 0)
    pdf.cell(95, 15, f"{stats_live['score']}%", 1, 1, 'C') # Tanpa kurung tren
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 10)
    pdf.cell(47, 8, f"Total Data: {stats_past['total']}", 1, 0, 'C')
    pdf.cell(48, 8, f"Positif: {stats_past['pos']} | Negatif: {stats_past['neg']}", 1, 0, 'C')
    pdf.cell(47, 8, f"Total Data: {stats_live['total']}", 1, 0, 'C')
    pdf.cell(48, 8, f"Positif: {stats_live['pos']} | Negatif: {stats_live['neg']}", 1, 1, 'C')
    pdf.ln(5)

    # 2. Diagnosis AI
    pdf.chapter_title("2. Diagnosis AI (Analisis Kualitatif)")
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(150, 0, 0)
    pdf.cell(0, 8, "Analisis Keluhan & Masalah:", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, clean_text(analysis_neg))
    pdf.ln(2)

    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 8, "Analisis Kekuatan & Apresiasi:", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, clean_text(analysis_pos))
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # 3. Visualisasi (Pastikan tidak tumpang tindih)
    pdf.chapter_title("3. Peta Kata Kunci (Visualisasi)")
    y_start = pdf.get_y()
    
    if wc_neg:
        pdf.set_xy(10, y_start)
        pdf.set_font('Arial', 'B', 9); pdf.cell(90, 8, "Topik Keluhan", 0, 1, 'C')
        # Gunakan Custom X/Y, jadi tidak pakai auto set_y
        pdf.add_chart(wc_neg, x=10, y=y_start+8, w=90, h=50) 
    
    if wc_pos:
        pdf.set_xy(105, y_start)
        pdf.set_font('Arial', 'B', 9); pdf.cell(90, 8, "Topik Apresiasi", 0, 1, 'C')
        pdf.add_chart(wc_pos, x=105, y=y_start+8, w=90, h=50)
    
    pdf.set_y(y_start + 65) # Manual set Y untuk konten berikutnya

    # 4. Sampel Ulasan
    pdf.add_page()
    pdf.chapter_title("4. Sampel Suara Pasien")
    
    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(200, 0, 0)
    pdf.cell(0, 10, "Sampel Keluhan:", 0, 1)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(80, 80, 80)
    
    if not samples['neg']:
        pdf.cell(0, 8, "- Belum ada data.", 0, 1)
    else:
        for review in samples['neg']:
            pdf.multi_cell(0, 6, f"- \"{clean_text(review)}\"")
            pdf.ln(2)
            
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 11)
    pdf.set_text_color(0, 128, 0)
    pdf.cell(0, 10, "Sampel Apresiasi:", 0, 1)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(80, 80, 80)
    
    if not samples['pos']:
        pdf.cell(0, 8, "- Belum ada data.", 0, 1)
    else:
        for review in samples['pos']:
            pdf.multi_cell(0, 6, f"- \"{clean_text(review)}\"")
            pdf.ln(2)

    return pdf.output(dest='S')