from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

# Tabel Riwayat Prediksi
class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    model_used = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    # Fitur feedback: User bisa menandai apakah prediksi ini benar/salah nanti
    is_correct = db.Column(db.Boolean, default=None, nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'text': self.text,
            'sentiment': self.sentiment,
            'confidence': round(self.confidence, 2),
            'model_used': self.model_used,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M')
        }