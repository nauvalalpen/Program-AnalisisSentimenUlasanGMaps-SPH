import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# --- IMPORT MODEL-MODEL KUAT ---
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier # <--- INI KUNCINYA

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import io
import base64
import re
from collections import Counter
import numpy as np

# --- 1. KAMUS SLANG (Bahasa Gaul -> Baku) ---
# Menyamakan persepsi mesin terhadap singkatan umum
SLANG_DICT = {
    'ga': 'tidak', 'gak': 'tidak', 'nggak': 'tidak', 'tdk': 'tidak',
    'bgs': 'bagus', 'bgt': 'banget', 'dlm': 'dalam', 'dr': 'dari',
    'krn': 'karena', 'jg': 'juga', 'tp': 'tapi', 'blm': 'belum',
    'sdh': 'sudah', 'dgn': 'dengan', 'thx': 'terima kasih',
    'sy': 'saya', 'aku': 'saya', 'kalo': 'kalau', 'klo': 'kalau',
    'org': 'orang', 'skt': 'sakit', 'rs': 'rumah sakit',
    'antri': 'antre', 'antrian': 'antrean', 'pelayanan': 'layanan'
}

STOPWORDS_ID = set([
    'dan', 'di', 'ke', 'dari', 'yang', 'pada', 'adalah', 'ini', 'itu', 'untuk', 
    'saya', 'kami', 'kita', 'kamu', 'dia', 'mereka', 'juga', 'dengan', 'ya', 'yuk'
])

class SentimentEngine:
    def __init__(self):
        self.model = None
        self.hospital_list = []
        self.metrics = {} 
        self.historical_data = {'positive': "", 'negative': "", 'df': None}
        self.learning_curve_img = None 

    def _normalize_slang(self, text):
        """Mengubah kata gaul menjadi baku"""
        if not isinstance(text, str): return str(text)
        words = text.split()
        normalized_words = [SLANG_DICT.get(w, w) for w in words]
        return " ".join(normalized_words)

    def train_model(self, excel_path):
        df = pd.read_excel(excel_path)
        self.hospital_list = sorted(df['Nama Rumah Sakit'].unique().tolist())
        self.historical_data['df'] = df

        # Normalisasi Slang dulu sebelum disimpan
        df['Komentar Bersih'] = df['Komentar Bersih'].astype(str).apply(self._normalize_slang)

        self.historical_data['positive'] = " ".join(df[df['Label'] == 'Positif']['Komentar Bersih'])
        self.historical_data['negative'] = " ".join(df[df['Label'] == 'Negatif']['Komentar Bersih'])

        X = df['Komentar Bersih']
        y = df['Label']
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y  # <--- INI KUNCINYA
        )

        # --- 2. DEFINISI OTAK-OTAK AI ---
        clf1 = MultinomialNB() # Bagus untuk teks pendek
        clf2 = LogisticRegression(random_state=1, max_iter=1000) # Bagus untuk klasifikasi biner
        clf3 = SVC(kernel='linear', probability=True) # Bagus untuk memisahkan data yang rumit

        # Gabungkan menjadi satu kekuatan (Voting)
        # 'soft' voting berarti mereka berunding berdasarkan probabilitas keyakinan
        eclf = VotingClassifier(estimators=[
            ('nb', clf1), 
            ('lr', clf2), 
            ('svc', clf3)
        ], voting='soft')

        # Masukkan ke dalam Pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words=list(STOPWORDS_ID), 
                ngram_range=(1, 2), # Unigram + Bigram
                max_features=5000
            )),
            ('ensemble', eclf)
        ])

        # Train Model Ensemble
        self.model.fit(X_train, y_train)

        # 3. Evaluasi
        y_pred = self.model.predict(X_test)
        
        self.metrics['accuracy'] = round(accuracy_score(y_test, y_pred) * 100, 2)
        self.metrics['precision'] = round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        self.metrics['recall'] = round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        self.metrics['f1_score'] = round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        
        # Confusion Matrix
        labels = ['Negatif', 'Positif']
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        self.metrics['tp'], self.metrics['tn'] = int(tp), int(tn)
        self.metrics['fp'], self.metrics['fn'] = int(fp), int(fn)

        # Cek Overfitting
        y_train_pred = self.model.predict(X_train)
        self.metrics['train_accuracy'] = round(accuracy_score(y_train, y_train_pred) * 100, 2)
        
        diff = self.metrics['train_accuracy'] - self.metrics['accuracy']
        if self.metrics['train_accuracy'] < 70:
            self.metrics['status'] = "Underfitting"
            self.metrics['advice'] = "Model kurang kompleks."
        elif diff > 12: # Kita perketat toleransi gap menjadi 12%
            self.metrics['status'] = "Overfitting"
            self.metrics['advice'] = "Model terlalu menghafal. Tambah variasi data training."
        else:
            self.metrics['status'] = "Good Fit (Ideal)"
            self.metrics['advice'] = "Keseimbangan sempurna antara hafalan dan pemahaman."

        # Generate Learning Curve
        self.learning_curve_img = self._create_learning_curve(self.model, X, y)
        
        print(f"Final Accuracy (Ensemble): {self.metrics['accuracy']}%")

    def _create_learning_curve(self, estimator, X, y):
        # Menggunakan Figure baru untuk menghindari tumpang tindih
        fig, ax = plt.subplots(figsize=(8, 6))
        
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 5)
        )
        
        train_scores_mean = np.mean(train_scores, axis=1) * 100
        test_scores_mean = np.mean(test_scores, axis=1) * 100

        ax.plot(train_sizes, train_scores_mean, 'o-', color="#ef4444", label="Training Score")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="#10b981", label="Validation Score")
        
        ax.set_title("Learning Curve: Ensemble Model Performance")
        ax.set_xlabel("Jumlah Data Training")
        ax.set_ylabel("Akurasi (%)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        img = io.BytesIO()
        plt.tight_layout()
        fig.savefig(img, format='png')
        plt.close(fig) # Tutup figure secara eksplisit
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')

    def predict(self, text):
        if not self.model: return "Error", 0
        # Normalisasi input user juga
        clean_text = self._normalize_slang(text)
        
        prediction = self.model.predict([clean_text])[0]
        proba = self.model.predict_proba([clean_text]).max() * 100
        return prediction, round(proba, 2)

    def generate_wordcloud(self, text_data, color_theme='viridis'):
        if not text_data or len(text_data.strip()) == 0: return None
        wc = WordCloud(width=800, height=400, background_color='white', colormap=color_theme, max_words=100).generate(text_data)
        
        img_obj = wc.to_image()
        img_buffer = io.BytesIO()
        img_obj.save(img_buffer, format='PNG')
        return base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    # Helper Analisis Teks tidak berubah
    def extract_top_keywords(self, text, top_n=5):
        if not text: return []
        words = re.findall(r'\w+', text.lower())
        stopwords = list(STOPWORDS_ID) + ['rs', 'rumah', 'sakit', 'padang', 'yang', 'dan', 'di', 'yg', 'ga', 'gak', 'nya']
        words = [w for w in words if w not in stopwords and len(w) > 3]
        return [word for word, count in Counter(words).most_common(top_n)]

    def generate_analysis(self, past_text, current_text, sentiment_type):
        past_keywords = set(self.extract_top_keywords(past_text))
        current_keywords = set(self.extract_top_keywords(current_text))
        
        if not current_keywords: return "Data terkini belum cukup."
        
        common = past_keywords.intersection(current_keywords)
        new_issues = current_keywords - past_keywords
        
        if sentiment_type == 'Negatif':
            if len(new_issues) > 0: return f"⚠️ Isu Baru: '{', '.join(list(new_issues)[:3])}'."
            elif len(common) >= 1: return f"🔄 Masalah Lama: '{', '.join(list(common)[:3])}' masih ada."
            else: return "✅ Perbaikan Signifikan."
        else:
            if len(new_issues) > 0: return f"📈 Peningkatan: '{', '.join(list(new_issues)[:3])}'."
            else: return f"🌟 Konsisten Bagus: '{', '.join(list(common)[:3])}'."
    
    # Fungsi generate_plot_bytes untuk PDF (diperlukan untuk app.py)
    def generate_plot_bytes(self, plot_type, data=None):
        img = io.BytesIO()
        plt.figure(figsize=(6, 4)) # Ukuran sedikit lebih kecil agar pas di PDF
        
        if plot_type == 'sentiment_dist':
            # Pie Chart
            labels = ['Positif', 'Negatif']
            # Cek jika data kosong untuk menghindari error pie chart
            if sum(data) == 0: data = [1, 0] 
            
            plt.pie(data, labels=labels, autopct='%1.1f%%', colors=['#198754', '#dc3545'], startangle=90)
            plt.title('Distribusi Sentimen (Live Data)')

        plt.tight_layout()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return img

ai_brain = SentimentEngine()