from fpdf import FPDF
import tempfile
import os
from datetime import datetime

class PDFReport(FPDF):
    def header(self):
        # Header Laporan Profesional
        self.set_font('Arial', 'B', 14)
        self.set_text_color(153, 27, 27) # Merah SPH
        self.cell(0, 8, 'LAPORAN ANALISIS SENTIMEN PASIEN', 0, 1, 'L')
        
        self.set_font('Arial', '', 10)
        self.set_text_color(80, 80, 80)
        tanggal = datetime.now().strftime("%d %B %Y, %H:%M")
        self.cell(0, 5, f'Rumah Sakit Semen Padang | Generated: {tanggal}', 0, 1, 'L')
        
        # Garis Merah Tebal
        self.set_draw_color(153, 27, 27)
        self.set_line_width(1)
        self.line(10, 25, 200, 25)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Halaman {self.page_no()} | Powered by SPH AI System', 0, 0, 'C')

def create_pdf_report(data):
    """
    Fungsi Layout PDF Profesional (Multi-Page).
    Data input berupa Dictionary lengkap dari ml_engine.
    """
    pdf = PDFReport()
    
    # --- HALAMAN 1: OVERVIEW & STATISTIK ---
    pdf.add_page()
    
    # Judul Besar
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "1. Ringkasan Eksekutif", 0, 1)
    
    # Kotak Statistik
    pdf.set_fill_color(240, 240, 240) # Abu muda
    pdf.set_font("Arial", '', 11)
    stats = data['stats']
    metrics = data['metrics']
    
    text_summary = (
        f"Analisis dilakukan terhadap {stats['total']} ulasan pasien dari Google Maps. "
        f"Hasil menunjukkan dominasi sentimen {(stats['pos']/stats['total']*100):.1f}% POSITIF. "
        f"Model AI '{data['best_model']}' dipilih sebagai algoritma terbaik dengan akurasi {metrics['accuracy']}%."
    )
    pdf.multi_cell(0, 7, text_summary, 0, 'J', False)
    pdf.ln(5)

    try:
        # Layout: Kiri Pie Chart, Kanan Bar Chart
        y_pos = pdf.get_y()
        
        # 1. PIE CHART
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_pie:
            tmp_pie.write(data['pie_bytes'])
            tmp_pie.close()
            pdf.image(tmp_pie.name, x=10, y=y_pos, w=90)
            os.unlink(tmp_pie.name)

        # 2. BAR CHART (Perbandingan Model)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_bar:
            tmp_bar.write(data['bar_bytes'])
            tmp_bar.close()
            pdf.image(tmp_bar.name, x=105, y=y_pos+10, w=95) # Sedikit turun agar center
            os.unlink(tmp_bar.name)
            
        pdf.ln(100) # Spasi ke bawah melewati gambar
        
    except Exception as e:
        pdf.cell(0, 10, f"Error Grafik Hal 1: {str(e)}", 1, 1)

    # --- HALAMAN 2: DEEP DIVE (WORDCLOUD) ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "2. Analisis Topik (Wordcloud)", 0, 1)
    pdf.set_font("Arial", '', 11)
    pdf.multi_cell(0, 7, "Visualisasi di bawah ini menampilkan kata-kata yang paling sering muncul dalam ulasan, membantu manajemen mengidentifikasi kekuatan (Positif) dan keluhan (Negatif).")
    pdf.ln(5)

    try:
        # Label Positif
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(34, 197, 94) # Hijau
        pdf.cell(0, 10, "TOPIC: ULASAN POSITIF", 0, 1)
        
        # Gambar WC Positif
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_wc_pos:
            tmp_wc_pos.write(data['wc_pos_bytes'])
            tmp_wc_pos.close()
            pdf.image(tmp_wc_pos.name, x=15, y=pdf.get_y(), w=180)
            os.unlink(tmp_wc_pos.name)
        pdf.ln(95)

        # Label Negatif
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(239, 68, 68) # Merah
        pdf.cell(0, 10, "TOPIC: ULASAN NEGATIF", 0, 1)
        
        # Gambar WC Negatif
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_wc_neg:
            tmp_wc_neg.write(data['wc_neg_bytes'])
            tmp_wc_neg.close()
            pdf.image(tmp_wc_neg.name, x=15, y=pdf.get_y(), w=180)
            os.unlink(tmp_wc_neg.name)
            
    except Exception as e:
        pdf.set_text_color(0)
        pdf.cell(0, 10, f"Error Grafik Hal 2: {str(e)}", 1, 1)

    # --- HALAMAN 3: EVALUASI TEKNIS & REKOMENDASI ---
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "3. Detail Teknis & Rekomendasi", 0, 1)
    
    # Confusion Matrix di Kiri
    y_final = pdf.get_y()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_cm:
            tmp_cm.write(data['cm_bytes'])
            tmp_cm.close()
            pdf.image(tmp_cm.name, x=55, y=y_final, w=100) # Center
            os.unlink(tmp_cm.name)
        pdf.ln(90)
    except:
        pass

    # Kotak Rekomendasi
    pdf.set_fill_color(255, 245, 245) # Merah muda sangat pudar
    pdf.set_draw_color(153, 27, 27)
    pdf.rect(10, pdf.get_y(), 190, 60, 'DF')
    
    pdf.set_xy(15, pdf.get_y()+5)
    pdf.set_font("Arial", 'B', 14)
    pdf.set_text_color(153, 27, 27)
    pdf.cell(0, 10, "REKOMENDASI STRATEGIS", 0, 1)
    
    pdf.set_font("Arial", '', 11)
    pdf.set_text_color(0, 0, 0)
    
    # Logika Rekomendasi
    neg_pct = (stats['neg'] / stats['total']) * 100
    
    points = []
    if neg_pct > 30:
        points.append("1. [URGENT] Lakukan audit pada layanan front-office & antrian.")
        points.append("2. Perhatikan kata kunci dominan di Wordcloud Negatif.")
        points.append("3. Adakan pelatihan 'Service Excellence' untuk staf.")
    elif neg_pct > 15:
        points.append("1. Tingkatkan kecepatan respon terhadap keluhan pasien.")
        points.append("2. Monitor waktu tunggu pasien di jam sibuk.")
        points.append("3. Pertahankan keramahan dokter yang sudah dinilai baik.")
    else:
        points.append("1. [EXCELLENT] Pertahankan standar layanan saat ini.")
        points.append("2. Gunakan testimoni positif untuk materi promosi di media sosial.")
        points.append("3. Berikan reward kepada tim medis atas kinerja yang baik.")
        
    for p in points:
        pdf.set_x(15)
        pdf.multi_cell(180, 7, p)
        
    return pdf.output(dest='S')