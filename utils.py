from fpdf import FPDF
import tempfile
import os
import base64

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Laporan Lengkap: Hospital Sentiment Analytics', 0, 1, 'C')
        self.line(10, 25, 200, 25)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(240, 240, 240) # Abu-abu muda
        self.cell(0, 10, f"  {title}", 0, 1, 'L', 1)
        self.ln(5)

    def add_image_from_base64(self, b64_str, x, y, w, h):
        """Helper untuk menaruh gambar di koordinat spesifik"""
        if not b64_str: 
            # Jika tidak ada gambar, tulis placeholder
            self.set_xy(x, y + h/2)
            self.set_font('Arial', 'I', 8)
            self.cell(w, 10, "[Data Belum Cukup]", 0, 0, 'C')
            return

        try:
            img_data = base64.b64decode(b64_str)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(img_data)
                tmp_path = tmp_file.name
            
            self.image(tmp_path, x=x, y=y, w=w, h=h)
            os.unlink(tmp_path)
        except:
            pass

def clean_text(text):
    if not isinstance(text, str): return str(text)
    return text.encode('latin-1', 'ignore').decode('latin-1')

def create_pdf_report(model_metrics, live_stats, learning_curve_b64, rs_specific_data=None):
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- HALAMAN 1: KESEHATAN MODEL (TETAP) ---
    pdf.chapter_title("1. Evaluasi Model Machine Learning")
    pdf.set_font('Arial', '', 11)
    
    status = clean_text(f"Akurasi Model: {model_metrics.get('accuracy', 0)}% ({model_metrics.get('status', 'N/A')})")
    advice = clean_text(f"Analisis: {model_metrics.get('advice', 'N/A')}")
    
    pdf.cell(0, 7, status, 0, 1)
    pdf.multi_cell(0, 7, advice)
    pdf.ln(5)
    
    if learning_curve_b64:
        # Tampilkan grafik di tengah
        pdf.add_image_from_base64(learning_curve_b64, x=55, y=pdf.get_y(), w=100, h=70)
        pdf.ln(75) # Pindah baris setelah gambar

    # --- HALAMAN 2: ANALISIS SPESIFIK RS ---
    if rs_specific_data:
        pdf.add_page()
        rs_name = clean_text(rs_specific_data.get('name', 'Rumah Sakit'))
        pdf.chapter_title(f"2. Analisis Spesifik: {rs_name}")
        
        # --- JUDUL KOLOM ---
        pdf.set_font('Arial', 'B', 10)
        y_start = pdf.get_y()
        
        # Header Kolom Kiri
        pdf.set_fill_color(100, 100, 100) # Abu-abu tua
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(10, y_start)
        pdf.cell(90, 8, "MASA LALU (Dataset)", 0, 0, 'C', 1)
        
        # Header Kolom Kanan
        pdf.set_fill_color(0, 102, 204) # Biru
        pdf.set_xy(110, y_start)
        pdf.cell(90, 8, "MASA SEKARANG (Live)", 0, 1, 'C', 1)
        
        pdf.set_text_color(0, 0, 0) # Reset hitam
        pdf.ln(2)

        # --- BARIS 1: KELUHAN (NEGATIF) ---
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(150, 0, 0) # Merah
        pdf.cell(0, 6, "Komparasi Keluhan (Negatif)", 0, 1, 'C')
        
        y_img_1 = pdf.get_y()
        # Gambar Kiri (Negatif Past)
        pdf.add_image_from_base64(rs_specific_data.get('wc_neg_past'), x=10, y=y_img_1, w=90, h=45)
        # Gambar Kanan (Negatif Live)
        pdf.add_image_from_base64(rs_specific_data.get('wc_neg_live'), x=110, y=y_img_1, w=90, h=45)
        
        pdf.set_y(y_img_1 + 50) # Pindah cursor ke bawah gambar

        # --- BARIS 2: KEKUATAN (POSITIF) ---
        pdf.set_font('Arial', 'B', 9)
        pdf.set_text_color(0, 100, 0) # Hijau
        pdf.cell(0, 6, "Komparasi Kekuatan (Positif)", 0, 1, 'C')

        y_img_2 = pdf.get_y()
        # Gambar Kiri (Positif Past)
        pdf.add_image_from_base64(rs_specific_data.get('wc_pos_past'), x=10, y=y_img_2, w=90, h=45)
        # Gambar Kanan (Positif Live)
        pdf.add_image_from_base64(rs_specific_data.get('wc_pos_live'), x=110, y=y_img_2, w=90, h=45)

        pdf.set_y(y_img_2 + 55) # Pindah cursor ke bawah gambar

        # --- BAGIAN 3: KESIMPULAN / EVALUASI TEKS ---
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Kesimpulan & Evaluasi AI", 0, 1, 'L')
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        # Analisis Negatif
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(150, 0, 0)
        pdf.cell(0, 6, "Analisis Keluhan:", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0, 0, 0)
        # Tampilkan teks analisis dari AI
        neg_text = clean_text(rs_specific_data.get('analysis_neg', '-'))
        pdf.multi_cell(0, 5, neg_text)
        pdf.ln(3)

        # Analisis Positif
        pdf.set_font('Arial', 'B', 10)
        pdf.set_text_color(0, 100, 0)
        pdf.cell(0, 6, "Analisis Kekuatan:", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0, 0, 0)
        # Tampilkan teks analisis dari AI
        pos_text = clean_text(rs_specific_data.get('analysis_pos', '-'))
        pdf.multi_cell(0, 5, pos_text)

    # Output bytes
    try:
        return pdf.output(dest='S').encode('latin-1', 'ignore')
    except:
        return pdf.output(dest='S').encode('latin-1', 'replace')