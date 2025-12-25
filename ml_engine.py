import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import base64
import re
from collections import Counter
import numpy as np

class SentimentEngine:
    def __init__(self):
        self.model = None
        self.hospital_list = []
        self.metrics = {} 
        self.historical_data = {'positive': "", 'negative': "", 'df': None}
        self.learning_curve_img = None # Simpan gambar learning curve

    def train_model(self, excel_path):
        df = pd.read_excel(excel_path)
        self.hospital_list = df['Nama Rumah Sakit'].unique().tolist()
        self.historical_data['df'] = df

        self.historical_data['positive'] = " ".join(df[df['Label'] == 'Positif']['Komentar Bersih'].astype(str))
        self.historical_data['negative'] = " ".join(df[df['Label'] == 'Negatif']['Komentar Bersih'].astype(str))

        X = df['Komentar Bersih'].astype(str)
        y = df['Label']
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        
        # 1. GENERATE LEARNING CURVE (GRAFIK OVERFITTING)
        self.learning_curve_img = self._create_learning_curve(self.model, X, y)

        # 2. TRAIN & EVALUATE
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        self.metrics['accuracy'] = round(accuracy_score(y_test, y_pred) * 100, 2)
        self.metrics['precision'] = round(precision_score(y_test, y_pred, average='macro') * 100, 2)
        self.metrics['recall'] = round(recall_score(y_test, y_pred, average='macro') * 100, 2)
        self.metrics['f1_score'] = round(f1_score(y_test, y_pred, average='macro') * 100, 2)
        
        # Cek Overfitting
        y_train_pred = self.model.predict(X_train)
        self.metrics['train_accuracy'] = round(accuracy_score(y_train, y_train_pred) * 100, 2)
        
        diff = self.metrics['train_accuracy'] - self.metrics['accuracy']
        if self.metrics['train_accuracy'] < 60:
            self.metrics['status'] = "Underfitting"
            self.metrics['advice'] = "Model terlalu sederhana atau data kurang."
        elif diff > 15:
            self.metrics['status'] = "Overfitting"
            self.metrics['advice'] = "Gap Training-Testing besar. Perlu tambah data."
        else:
            self.metrics['status'] = "Good Fit (Ideal)"
            self.metrics['advice'] = "Performa seimbang."

        # Final Train
        self.model.fit(X, y)
        print(f"Model Trained. Status: {self.metrics['status']}")

    def _create_learning_curve(self, estimator, X, y):
        """Membuat Grafik Learning Curve (Training vs Validation Score)"""
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        
        train_scores_mean = np.mean(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100

        plt.figure(figsize=(7, 5))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
        
        plt.title("Learning Curve: Deteksi Overfitting/Underfitting")
        plt.xlabel("Jumlah Data Training")
        plt.ylabel("Akurasi (%)")
        plt.legend(loc="best")
        plt.grid()
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')

    def predict(self, text):
        if not self.model: return "Error", 0
        prediction = self.model.predict([text])[0]
        proba = self.model.predict_proba([text]).max() * 100
        return prediction, round(proba, 2)

    def generate_wordcloud(self, text_data, color_theme='viridis'):
        if not text_data or len(text_data.strip()) == 0: return None
        wc = WordCloud(width=800, height=400, background_color='white', colormap=color_theme).generate(text_data)
        img = io.BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plt.close()
        return base64.b64encode(img.getvalue()).decode('utf-8')

    # Helper untuk Analisis Teks (Comparison)
    def extract_top_keywords(self, text, top_n=5):
        if not text: return []
        words = re.findall(r'\w+', text.lower())
        stopwords = ['dan', 'yang', 'di', 'ini', 'itu', 'untuk', 'saya', 'tidak', 'yg', 'ke', 'ada', 'rs', 'rumah', 'sakit']
        words = [w for w in words if w not in stopwords and len(w) > 3]
        return [word for word, count in Counter(words).most_common(top_n)]

    def generate_analysis(self, past_text, current_text, sentiment_type):
        past_keywords = set(self.extract_top_keywords(past_text))
        current_keywords = set(self.extract_top_keywords(current_text))
        if not current_keywords: return "Data terkini belum cukup."
        common = past_keywords.intersection(current_keywords)
        
        if sentiment_type == 'Negatif':
            if len(common) >= 2: return f"⚠️ Masalah Berulang: {', '.join(common)} masih dikeluhkan."
            elif len(current_keywords) > 0: return f"🔄 Masalah Baru: {', '.join(current_keywords)}."
            else: return "✅ Perbaikan: Tidak ada keluhan dominan."
        else:
            if len(common) >= 2: return f"🌟 Konsistensi: {', '.join(common)} tetap bagus."
            else: return f"📈 Peningkatan: {', '.join(current_keywords)}."

ai_brain = SentimentEngine()