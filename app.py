import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from models import db, PredictionHistory
from ml_engine import ai_engine
from flask import send_file
import io
import utils # Import modul utils yang baru kita buat

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sph_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

db.init_app(app)

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index(): return render_template('index.html')

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

@app.route('/history')
def history():
    data = PredictionHistory.query.order_by(PredictionHistory.id.desc()).all()
    return render_template('history.html', history=data)

# --- API UPDATED ---

@app.route('/api/upload_train', methods=['POST'])
def upload_train():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'})

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Train 3 Model Sekaligus
        success, result = ai_engine.train(filepath)
        
        if success:
            # Generate 2 Wordcloud (Positif & Negatif)
            wc_images = ai_engine.generate_dual_wordclouds(filepath)
            return jsonify({
                'status': 'success', 
                'evaluation': result['metrics'], # Hasil Akurasi 3 Model
                'stats': result['data_stats'],   # Total Data
                'wordcloud': wc_images           # Gambar WC
            })
        else:
            return jsonify({'status': 'error', 'message': result})

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    # Ambil model yang dipilih user dari dropdown (default NB)
    model_choice = data.get('model', 'Naive Bayes') 
    
    if not ai_engine.is_trained:
        return jsonify({'status': 'error', 'message': 'Model belum dilatih!'})

    # Prediksi menggunakan model pilihan
    result = ai_engine.predict(text, model_name=model_choice)
    
    # Simpan ke DB
    new_entry = PredictionHistory(
        text=text,
        sentiment=result['sentiment'],
        confidence=result['confidence'],
        model_used=result['model'] # Simpan nama model yang dipakai
    )
    db.session.add(new_entry)
    db.session.commit()

    return jsonify({'status': 'success', 'result': result})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    db.session.query(PredictionHistory).delete()
    db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/api/download_report')
def download_report():
    if not ai_engine.is_trained:
        return "Model belum dilatih! Silakan upload data dulu.", 400
        
    # Ambil file terakhir yang diupload untuk dianalisis ulang grafiknya
    # (Cara gampangnya: kita cari file excel di folder uploads)
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not files:
        return "File data tidak ditemukan.", 404
        
    # Ambil file excel terakhir
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], files[0])
    
    # 1. Generate Data & Grafik dari ML Engine
    data, err = ai_engine.generate_report_assets(filepath)
    if err:
        return f"Error: {err}", 500
        
    # 2. Buat PDF menggunakan Utils
    pdf_bytes = utils.create_pdf_report(
        stats=data['stats'],
        metrics=data['metrics'],
        best_model_name=data['best_model'],
        img_pie=data['pie_bytes'],
        img_cm=data['cm_bytes']
    )
    
    # 3. Kirim File ke Browser
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype='application/pdf',
        as_attachment=True,
        download_name='Laporan_Analisis_SPH.pdf'
    )

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)