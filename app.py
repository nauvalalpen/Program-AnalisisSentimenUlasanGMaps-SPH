from flask import Flask, render_template, request, redirect, url_for, send_file, make_response
from models import db, ReviewLog
from ml_engine import ai_brain
import os
import io
import utils 
from datetime import datetime, timedelta
import pytz # Library Timezone
from fpdf import FPDF # Library PDF
from flask import jsonify
from sqlalchemy import func
import base64

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///hospital_reviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# Konfigurasi Timezone WIB
WIB = pytz.timezone('Asia/Jakarta')

# Custom Filter untuk Template (Convert UTC to WIB)
@app.template_filter('datetime_wib')
def format_datetime(value):
    if value is None: return ""
    # Asumsikan di DB disimpan sebagai UTC, convert ke WIB
    return value.replace(tzinfo=pytz.utc).astimezone(WIB).strftime('%d %b %Y, %H:%M WIB')

with app.app_context():
    db.create_all()
    if os.path.exists('dataset/dataset_gabungan_5rs.xlsx'):
        ai_brain.train_model('dataset/dataset_gabungan_5rs.xlsx')

@app.route('/')
def landing_page():
    # 1. Hitung Total Laporan Masuk
    total_reports = ReviewLog.query.count()
    
    # 2. Hitung Jumlah Rumah Sakit yang Terdaftar (Dari inputan user di DB)
    # Menggunakan distinct untuk menghitung nama RS unik
    total_hospitals = db.session.query(func.count(func.distinct(ReviewLog.hospital_name))).scalar()
    
    # Fallback jika DB kosong, ambil dari list Excel
    if total_hospitals == 0 and ai_brain.hospital_list:
        total_hospitals = len(ai_brain.hospital_list)

    # 3. Ambil Akurasi Model AI
    accuracy = ai_brain.metrics.get('accuracy', 0)
    
    # 4. Hitung Laporan Positif (Untuk Card di bawah)
    positive_reports = ReviewLog.query.filter_by(sentiment='Positif').count()
    
    # 5. Jumlah Partisipan (Kita asumsikan 1 laporan = 1 partisipan unik untuk simplifikasi, 
    # atau bisa hitung total reports saja)
    participants = total_reports 

    # Kirim data ke template
    return render_template('landing.html', 
                           total_reports=total_reports,
                           total_hospitals=total_hospitals,
                           accuracy=accuracy,
                           positive_reports=positive_reports,
                           participants=participants)

@app.route('/lapor', methods=['GET', 'POST'])
def input_page():
    """Halaman User melapor pengalaman (Dipindah dari route /)"""
    if request.method == 'POST':
        rs_name = request.form['hospital']
        review = request.form['review']

        # 1. Analisis dengan AI
        sentimen, conf = ai_brain.predict(review)

        # 2. Simpan ke Database
        new_log = ReviewLog(
            hospital_name=rs_name,
            review_text=review,
            sentiment=sentimen,
            confidence=conf,
            timestamp=datetime.utcnow() 
        )
        db.session.add(new_log)
        db.session.commit()
        
        return render_template('input.html', 
                               hospitals=ai_brain.hospital_list, 
                               success=True, 
                               result={'rs': rs_name, 'sentimen': sentimen})

    return render_template('input.html', hospitals=ai_brain.hospital_list)

@app.route('/dashboard')
def dashboard_page():
    # 1. Ambil Filter dari URL (?timeframe=week)
    timeframe = request.args.get('timeframe', 'all')
    
    # 2. Tentukan Batas Waktu
    now = datetime.utcnow()
    if timeframe == 'day': cutoff = now - timedelta(days=1)
    elif timeframe == 'week': cutoff = now - timedelta(weeks=1)
    elif timeframe == 'month': cutoff = now - timedelta(days=30)
    elif timeframe == 'year': cutoff = now - timedelta(days=365)
    else: cutoff = datetime.min # Semua data

    stats = {}
    
    # 3. Query dengan Filter Waktu
    # Ambil semua data sesuai filter dulu untuk efisiensi WordCloud
    filtered_logs = ReviewLog.query.filter(ReviewLog.timestamp >= cutoff).all()
    
    # Kumpulkan teks untuk WordCloud Global (Developer Feature)
    all_negative_text = " ".join([log.review_text for log in filtered_logs if log.sentiment == 'Negatif'])
    wordcloud_img = ai_brain.generate_wordcloud(all_negative_text)

    for rs in ai_brain.hospital_list:
        # Filter list di memori (lebih cepat daripada query berulang untuk data kecil)
        rs_logs = [log for log in filtered_logs if log.hospital_name == rs]
        total = len(rs_logs)
        
        if total > 0:
            pos = len([log for log in rs_logs if log.sentiment == 'Positif'])
            neg = len([log for log in rs_logs if log.sentiment == 'Negatif'])
            score = round((pos / total) * 100)
            stats[rs] = {'score': score, 'total': total, 'pos': pos, 'neg': neg}
        else:
            stats[rs] = {'score': 0, 'total': 0, 'pos': 0, 'neg': 0}

    sorted_stats = dict(sorted(stats.items(), key=lambda item: item[1]['score'], reverse=True))

    return render_template('dashboard.html', 
                           stats=sorted_stats, 
                           timeframe=timeframe) # Developer Info

@app.route('/history')
def history_page():
    logs = ReviewLog.query.order_by(ReviewLog.timestamp.desc()).all()
    return render_template('history.html', logs=logs)

# --- FITUR DEVELOPER (Delete & Report) ---

@app.route('/delete/<int:id>')
def delete_log(id):
    log = ReviewLog.query.get_or_404(id)
    db.session.delete(log)
    db.session.commit()
    return redirect(url_for('history_page'))

@app.route('/developer')
def developer_page():
    # PERBAIKAN: Ambil data dari DATASET (Excel), bukan dari Database Live
    
    # Ambil teks yang sudah disimpan di memori saat training (ml_engine.py)
    text_positif_dataset = ai_brain.historical_data['positive']
    text_negatif_dataset = ai_brain.historical_data['negative']
    
    # Generate Wordcloud dari Dataset
    wc_global_pos = ai_brain.generate_wordcloud(text_positif_dataset, 'Greens')
    wc_global_neg = ai_brain.generate_wordcloud(text_negatif_dataset, 'Reds')
    
    return render_template('developer.html', 
                           metrics=ai_brain.metrics,
                           hospitals=ai_brain.hospital_list,
                           wc_pos=wc_global_pos,
                           wc_neg=wc_global_neg,
                           learning_curve=ai_brain.learning_curve_img)
    
@app.route('/api/get_hospital_analysis', methods=['POST'])
def get_hospital_analysis():
    rs_name = request.json.get('hospital')
    
    # 1. Ambil Data MASA LALU (Dari Excel/Training)
    # Filter dataframe yang disimpan di ml_engine
    df_past = ai_brain.historical_data['df']
    df_rs_past = df_past[df_past['Nama Rumah Sakit'] == rs_name]
    
    text_pos_past = " ".join(df_rs_past[df_rs_past['Label'] == 'Positif']['Komentar Bersih'].astype(str))
    text_neg_past = " ".join(df_rs_past[df_rs_past['Label'] == 'Negatif']['Komentar Bersih'].astype(str))

    # 2. Ambil Data MASA SEKARANG (Dari Database SQLite)
    logs = ReviewLog.query.filter_by(hospital_name=rs_name).all()
    text_pos_live = " ".join([log.review_text for log in logs if log.sentiment == 'Positif'])
    text_neg_live = " ".join([log.review_text for log in logs if log.sentiment == 'Negatif'])
    
    # 3. Generate Wordclouds (4 Gambar: Past +, Past -, Live +, Live -)
    wc_pos_past = ai_brain.generate_wordcloud(text_pos_past, 'Greens')
    wc_neg_past = ai_brain.generate_wordcloud(text_neg_past, 'Reds')
    wc_pos_live = ai_brain.generate_wordcloud(text_pos_live, 'Greens')
    wc_neg_live = ai_brain.generate_wordcloud(text_neg_live, 'Reds')

    # 4. Generate AI Analysis Text
    analysis_neg = ai_brain.generate_analysis(text_neg_past, text_neg_live, 'Negatif')
    analysis_pos = ai_brain.generate_analysis(text_pos_past, text_pos_live, 'Positif')

    return jsonify({
        'wc_pos_past': wc_pos_past,
        'wc_neg_past': wc_neg_past,
        'wc_pos_live': wc_pos_live,
        'wc_neg_live': wc_neg_live,
        'analysis_neg': analysis_neg,
        'analysis_pos': analysis_pos
    })

# API untuk Dropdown Wordcloud (AJAX)
@app.route('/api/get_wordcloud', methods=['POST'])
def get_hospital_wordcloud():
    hospital_name = request.json.get('hospital')
    
    # PERBAIKAN: Filter data dari DataFrame DATASET (Excel), bukan Database
    df = ai_brain.historical_data['df']
    
    if df is not None:
        # Filter DataFrame berdasarkan Nama RS
        df_rs = df[df['Nama Rumah Sakit'] == hospital_name]
        
        # Ambil kolom komentar bersih berdasarkan label
        text_pos = " ".join(df_rs[df_rs['Label'] == 'Positif']['Komentar Bersih'].astype(str))
        text_neg = " ".join(df_rs[df_rs['Label'] == 'Negatif']['Komentar Bersih'].astype(str))
        
        # Generate Gambar
        wc_pos = ai_brain.generate_wordcloud(text_pos, 'Greens')
        wc_neg = ai_brain.generate_wordcloud(text_neg, 'Reds')
    else:
        wc_pos, wc_neg = None, None
    
    return jsonify({
        'wc_pos': wc_pos,
        'wc_neg': wc_neg
    })
    
@app.route('/api/download_report')
def download_report():
    if not ai_brain.model:
        return "Model belum dilatih! Silakan upload data training dulu.", 400
        
    # 1. Ambil Data Log (Live Data)
    logs = ReviewLog.query.all()
    
    # 2. Statistik Global
    total = len(logs)
    pos = len([l for l in logs if l.sentiment == 'Positif'])
    neg = len([l for l in logs if l.sentiment == 'Negatif'])
    live_stats = {'total': total, 'pos': pos, 'neg': neg}

    # 3. Generate Pie Chart (Distribusi)
    pie_chart_img = None
    if total > 0:
        pie_chart_img = ai_brain.generate_plot_bytes('sentiment_dist', [pos, neg])
        pie_chart_img = base64.b64encode(pie_chart_img.getvalue()).decode('utf-8')

    # 4. Generate Wordcloud Global
    text_pos = ai_brain.historical_data['positive']
    text_neg = ai_brain.historical_data['negative']
    wc_pos_img = ai_brain.generate_wordcloud(text_pos, 'Greens')
    wc_neg_img = ai_brain.generate_wordcloud(text_neg, 'Reds')

    # 5. Data Peringkat Rumah Sakit
    hospital_ranks = []
    for rs in ai_brain.hospital_list:
        rs_logs = [l for l in logs if l.hospital_name == rs]
        rs_total = len(rs_logs)
        if rs_total > 0:
            rs_pos = len([l for l in rs_logs if l.sentiment == 'Positif'])
            score = round((rs_pos / rs_total) * 100)
            hospital_ranks.append({
                'name': rs, 'total': rs_total, 'pos': rs_pos, 'score': score
            })
        else:
            hospital_ranks.append({'name': rs, 'total': 0, 'pos': 0, 'score': 0})
    
    hospital_ranks = sorted(hospital_ranks, key=lambda x: x['score'], reverse=True)

    # 6. Generate PDF (Raw Output)
    raw_pdf = utils.create_global_report(
        ai_brain.metrics,
        live_stats,
        ai_brain.learning_curve_img,
        wc_pos_img,
        wc_neg_img,
        hospital_ranks,
        pie_chart_img
    )

    # === PERBAIKAN DI SINI (Konversi String ke Bytes) ===
    if isinstance(raw_pdf, str):
        # Jika outputnya string, encode ke latin-1
        pdf_bytes = raw_pdf.encode('latin-1')
    elif isinstance(raw_pdf, bytearray):
        # Jika outputnya bytearray, ubah ke bytes
        pdf_bytes = bytes(raw_pdf)
    else:
        # Jika sudah bytes, biarkan
        pdf_bytes = raw_pdf

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='Laporan_Lengkap_SPH.pdf'
    )
    
@app.route('/api/download_report_specific/<rs_name>')
def download_report_specific(rs_name):
    if not ai_brain.model: return "Model belum dilatih!", 400

    # 1. DATA MASA LALU (Dataset Excel)
    df_past = ai_brain.historical_data['df']
    df_rs_past = df_past[df_past['Nama Rumah Sakit'] == rs_name]
    
    past_total = len(df_rs_past)
    past_pos = len(df_rs_past[df_rs_past['Label'] == 'Positif'])
    past_neg = len(df_rs_past[df_rs_past['Label'] == 'Negatif'])
    past_score = round((past_pos / past_total * 100), 1) if past_total > 0 else 0

    text_neg_past = " ".join(df_rs_past[df_rs_past['Label'] == 'Negatif']['Komentar Bersih'].astype(str))
    text_pos_past = " ".join(df_rs_past[df_rs_past['Label'] == 'Positif']['Komentar Bersih'].astype(str))

    # 2. DATA MASA SEKARANG (Database SQLite)
    logs = ReviewLog.query.filter_by(hospital_name=rs_name).order_by(ReviewLog.timestamp.desc()).all()
    
    live_total = len(logs)
    live_pos = len([l for l in logs if l.sentiment == 'Positif'])
    live_neg = len([l for l in logs if l.sentiment == 'Negatif'])
    live_score = round((live_pos / live_total * 100), 1) if live_total > 0 else 0
    
    text_neg_live = " ".join([log.review_text for log in logs if log.sentiment == 'Negatif'])
    text_pos_live = " ".join([log.review_text for log in logs if log.sentiment == 'Positif'])

    # 3. CONTOH ULASAN (Bukti Nyata)
    # Ambil 3 ulasan terbaru untuk masing-masing sentimen
    sample_reviews = {
        'pos': [l.review_text for l in logs if l.sentiment == 'Positif'][:3],
        'neg': [l.review_text for l in logs if l.sentiment == 'Negatif'][:3]
    }

    # 4. Generate Wordclouds (Prioritas Live, fallback ke Past jika kosong)
    wc_text_neg = text_neg_live if len(text_neg_live.strip()) > 5 else text_neg_past
    wc_text_pos = text_pos_live if len(text_pos_live.strip()) > 5 else text_pos_past
    
    wc_neg = ai_brain.generate_wordcloud(wc_text_neg, 'Reds')
    wc_pos = ai_brain.generate_wordcloud(wc_text_pos, 'Greens')

    # 5. Analisis Naratif
    analysis_neg = ai_brain.generate_analysis(text_neg_past, text_neg_live, 'Negatif')
    analysis_pos = ai_brain.generate_analysis(text_pos_past, text_pos_live, 'Positif')

    # 6. PANGGIL PDF GENERATOR
    pdf_output = utils.create_specific_report(
        rs_name=rs_name,
        stats_past={'total': past_total, 'pos': past_pos, 'neg': past_neg, 'score': past_score},
        stats_live={'total': live_total, 'pos': live_pos, 'neg': live_neg, 'score': live_score},
        analysis_neg=analysis_neg,
        analysis_pos=analysis_pos,
        wc_neg=wc_neg,
        wc_pos=wc_pos,
        samples=sample_reviews
    )

    if isinstance(pdf_output, str):
        pdf_bytes = pdf_output.encode('latin-1')
    else:
        pdf_bytes = bytes(pdf_output)
    
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'Laporan_Audit_{rs_name}.pdf'
    )

if __name__ == '__main__':
    app.run(debug=True)