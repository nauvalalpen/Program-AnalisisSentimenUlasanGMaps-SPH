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

    def add_chart(self, b64_str, h=80):
        if not b64_str: return
        try:
            img_data = base64.b64decode(b64_str)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_data)
                tmp_path = tmp_file.name
            
            # Center image
            x_pos = (210 - 140) / 2 # A4 width 210, img width 140
            self.image(tmp_path, x=x_pos, w=140, h=h)
            os.unlink(tmp_path)
            self.ln(5)
        except:
            self.cell(0, 10, "[Gambar Tidak Tersedia]", 0, 1)

def clean_text(text):
    if not isinstance(text, str): return str(text)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_global_report(metrics, stats, learning_curve, wc_pos, wc_neg, hospital_ranks, pie_chart):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- HALAMAN 1: RINGKASAN EKSEKUTIF ---
    pdf.add_page()
    
    # Judul Besar
    pdf.set_font('Arial', 'B', 24)
    pdf.cell(0, 20, "Laporan Analisis Sentimen", 0, 1, 'C')
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f"Tanggal Cetak: {datetime.now().strftime('%d %B %Y')}", 0, 1, 'C')
    pdf.ln(10)
    
    pdf.chapter_title("1. Ringkasan Eksekutif")
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 7, 
        f"Laporan ini menyajikan hasil analisis otomatis terhadap ulasan pelayanan rumah sakit. "
        f"Data dikumpulkan secara real-time dan dianalisis menggunakan Artificial Intelligence."
    )
    pdf.ln(5)

    # Kotak Statistik
    pdf.set_fill_color(245, 245, 245)
    pdf.rect(10, pdf.get_y(), 190, 30, 'F')
    pdf.set_y(pdf.get_y() + 5)
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(63, 10, "Total Laporan", 0, 0, 'C')
    pdf.cell(63, 10, "Sentimen Positif", 0, 0, 'C')
    pdf.cell(63, 10, "Sentimen Negatif", 0, 1, 'C')
    
    pdf.set_font('Arial', '', 16)
    pdf.set_text_color(0, 0, 255)
    pdf.cell(63, 10, str(stats['total']), 0, 0, 'C')
    pdf.set_text_color(0, 128, 0)
    pdf.cell(63, 10, str(stats['pos']), 0, 0, 'C')
    pdf.set_text_color(200, 0, 0)
    pdf.cell(63, 10, str(stats['neg']), 0, 1, 'C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)

    # Grafik Pie Chart Distribusi
    if pie_chart:
        pdf.section_subtitle("Distribusi Sentimen Keseluruhan:")
        pdf.add_chart(pie_chart, h=90)

    # --- HALAMAN 2: EVALUASI TEKNIS (AI) ---
    pdf.add_page()
    pdf.chapter_title("2. Evaluasi Model AI (Technical Audit)")
    
    # Tabel Metrik
    pdf.section_subtitle("Metrik Performa Model (Naive Bayes Ensemble):")
    pdf.set_font('Arial', '', 10)
    
    # Header Tabel
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(45, 10, "Metrik", 1, 0, 'C', 1)
    pdf.cell(45, 10, "Nilai (%)", 1, 0, 'C', 1)
    pdf.cell(100, 10, "Keterangan", 1, 1, 'C', 1)
    
    # Isi Tabel
    pdf.cell(45, 10, "Akurasi (Accuracy)", 1, 0)
    pdf.cell(45, 10, f"{metrics['accuracy']}%", 1, 0, 'C')
    pdf.cell(100, 10, "Ketepatan prediksi keseluruhan", 1, 1)
    
    pdf.cell(45, 10, "Training Score", 1, 0)
    pdf.cell(45, 10, f"{metrics['train_accuracy']}%", 1, 0, 'C')
    pdf.cell(100, 10, "Kemampuan model mempelajari data latih", 1, 1)
    
    pdf.cell(45, 10, "F1-Score", 1, 0)
    pdf.cell(45, 10, f"{metrics['f1_score']}%", 1, 0, 'C')
    pdf.cell(100, 10, "Keseimbangan Precision & Recall", 1, 1)
    pdf.ln(10)

    # Learning Curve
    pdf.section_subtitle("Grafik Deteksi Overfitting (Learning Curve):")
    status = clean_text(metrics.get('status', ''))
    advice = clean_text(metrics.get('advice', ''))
    
    pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, f"Status Model: {status}\nAnalisis: {advice}")
    pdf.ln(2)
    
    if learning_curve:
        pdf.add_chart(learning_curve, h=80)

    # --- HALAMAN 3: ANALISIS TEKS GLOBAL ---
    pdf.add_page()
    pdf.chapter_title("3. Analisis Teks Global")
    
    pdf.section_subtitle("Peta Kata Keluhan (Sentimen Negatif):")
    if wc_neg:
        pdf.add_chart(wc_neg, h=70)
    else:
        pdf.cell(0, 10, "Belum cukup data negatif.", 0, 1, 'C')
        
    pdf.ln(5)
    pdf.section_subtitle("Peta Kata Apresiasi (Sentimen Positif):")
    if wc_pos:
        pdf.add_chart(wc_pos, h=70)
    else:
        pdf.cell(0, 10, "Belum cukup data positif.", 0, 1, 'C')

   # --- HALAMAN 4: PERINGKAT RUMAH SAKIT ---
    pdf.add_page()
    pdf.chapter_title("4. Rincian Performa Rumah Sakit")
    
    # 1. SETUP HEADER TABEL
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(220, 230, 240)
    
    # Total lebar halaman A4 (Portrait) = ~190mm (margin kiri kanan 10mm)
    # Kita sesuaikan lebar kolom agar muat
    # No (10) + Nama (70) + Total (25) + Pos (25) + Neg (25) + Skor (35) = 190
    
    pdf.cell(10, 10, "No", 1, 0, 'C', 1)
    pdf.cell(70, 10, "Nama Rumah Sakit", 1, 0, 'C', 1)
    pdf.cell(25, 10, "Total", 1, 0, 'C', 1)
    pdf.cell(25, 10, "Positif", 1, 0, 'C', 1)
    pdf.cell(25, 10, "Negatif", 1, 0, 'C', 1) # <--- KOLOM BARU
    pdf.cell(35, 10, "Kepuasan", 1, 1, 'C', 1) # Pindah baris (1)
    
    # 2. ISI DATA TABEL
    pdf.set_font('Arial', '', 10)
    
    for i, rs in enumerate(hospital_ranks, 1):
        name = clean_text(rs['name'])
        # Potong nama jika terlalu panjang agar tidak merusak tabel
        if len(name) > 30: name = name[:27] + "..."
            
        # Hitung jumlah negatif (Total - Positif)
        # Atau jika 'neg' sudah dikirim dari app.py, pakai rs['neg']
        # Asumsi di app.py kita belum kirim key 'neg', kita hitung manual di sini:
        neg_count = rs['total'] - rs['pos'] 
        
        pdf.cell(10, 10, str(i), 1, 0, 'C')
        pdf.cell(70, 10, f" {name}", 1, 0, 'L')
        pdf.cell(25, 10, str(rs['total']), 1, 0, 'C')
        
        # Warna teks hijau untuk positif
        pdf.set_text_color(0, 100, 0)
        pdf.cell(25, 10, str(rs['pos']), 1, 0, 'C')
        
        # Warna teks merah untuk negatif (KOLOM BARU)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(25, 10, str(neg_count), 1, 0, 'C')
        
        # Reset warna hitam untuk skor
        pdf.set_text_color(0, 0, 0)
        score = rs['score']
        pdf.cell(35, 10, f"{score}%", 1, 1, 'C')

    return pdf.output(dest='S')

def create_specific_report(rs_name, analysis_neg, analysis_pos, wc_neg, wc_pos, wc_neg_past, wc_pos_past):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    clean_rs_name = clean_text(rs_name)
    
    # Judul
    pdf.set_font('Arial', 'B', 20)
    pdf.multi_cell(0, 10, f"Laporan Evaluasi: {clean_rs_name}", 0, 'C')
    pdf.ln(10)
    
    # Bagian 1: Analisis Naratif
    pdf.chapter_title("1. Diagnosis AI (Kesimpulan)")
    
    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(150, 0, 0)
    pdf.cell(0, 8, "Analisis Keluhan:", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, clean_text(analysis_neg))
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 10)
    pdf.set_text_color(0, 100, 0)
    pdf.cell(0, 8, "Analisis Kekuatan:", 0, 1)
    pdf.set_font('Arial', '', 10)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 6, clean_text(analysis_pos))
    pdf.ln(10)
    
    # Bagian 2: Visualisasi (Hanya yang Live/Terkini agar fokus)
    pdf.set_text_color(0, 0, 0)
    pdf.chapter_title("2. Visualisasi Kata Kunci (Terkini)")
    
    if wc_neg:
        pdf.section_subtitle("Peta Keluhan Saat Ini:")
        pdf.add_chart(wc_neg, h=70)
        
    if wc_pos:
        pdf.section_subtitle("Peta Kekuatan Saat Ini:")
        pdf.add_chart(wc_pos, h=70)
        
    return pdf.output(dest='S')
    