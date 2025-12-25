from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ReviewLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    hospital_name = db.Column(db.String(100), nullable=False) # Nama RS pilihan user
    review_text = db.Column(db.Text, nullable=False)          # Komentar user
    sentiment = db.Column(db.String(20), nullable=False)      # Hasil Prediksi AI (Positif/Negatif)
    confidence = db.Column(db.Float, nullable=False)          # Tingkat keyakinan AI (%)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow) # Waktu lapor

    def to_dict(self):
        return {
            'rs': self.hospital_name,
            'text': self.review_text,
            'sentiment': self.sentiment,
            'date': self.timestamp.strftime('%Y-%m-%d %H:%M')
        }