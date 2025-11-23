import pandas as pd
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class SentimentModel:
    def __init__(self):
        self.models = {} # Menyimpan semua model
        self.active_model_name = "Naive Bayes"
        self.vectorizer = None
        self.is_trained = False
        self.results = {} # Menyimpan skor evaluasi

    def train(self, filepath):
        try:
            df = pd.read_excel(filepath)
            if 'Komentar Bersih' not in df.columns or 'Label' not in df.columns:
                return False, "Format Excel salah. Kolom harus: 'Komentar Bersih' & 'Label'."

            X = df['Komentar Bersih'].astype(str)
            y = df['Label']

            # TF-IDF
            self.vectorizer = TfidfVectorizer(max_features=2000)
            X_vec = self.vectorizer.fit_transform(X)

            # Split Data
            X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

            # Definisi 3 Model
            candidates = {
                "Naive Bayes": MultinomialNB(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVM": SVC(kernel='linear', probability=True)
            }

            self.results = {}
            self.models = {}

            # Training Loop
            for name, model in candidates.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Hitung Metrik Lengkap
                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, pos_label='Positif', average='weighted')
                rec = recall_score(y_test, preds, pos_label='Positif', average='weighted')
                f1 = f1_score(y_test, preds, pos_label='Positif', average='weighted')

                self.results[name] = {
                    'accuracy': round(acc * 100, 2),
                    'precision': round(prec * 100, 2),
                    'recall': round(rec * 100, 2),
                    'f1_score': round(f1 * 100, 2)
                }
                self.models[name] = model

            self.is_trained = True
            
            # Statistik Data
            stats = {
                'total': len(df),
                'pos': len(df[df['Label'] == 'Positif']),
                'neg': len(df[df['Label'] == 'Negatif'])
            }
            
            return True, {'metrics': self.results, 'data_stats': stats}

        except Exception as e:
            return False, str(e)

    def predict(self, text, model_name="Naive Bayes"):
        if not self.is_trained:
            return None
        
        # Gunakan model yang dipilih user (default Naive Bayes)
        model = self.models.get(model_name, self.models["Naive Bayes"])
        
        vec = self.vectorizer.transform([text])
        pred = model.predict(vec)[0]
        proba = np.max(model.predict_proba(vec)) * 100
        
        return {'sentiment': pred, 'confidence': proba, 'model': model_name}

    def _create_wc_image(self, text, colormap):
        """Helper internal untuk bikin gambar wordcloud"""
        if not text.strip(): return None
        wc = WordCloud(width=800, height=400, background_color='white', colormap=colormap).generate(text)
        img = io.BytesIO()
        wc.to_image().save(img, format='PNG')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf-8')

    def generate_dual_wordclouds(self, filepath):
        df = pd.read_excel(filepath)
        
        # Pisahkan teks Positif dan Negatif
        text_pos = " ".join(df[df['Label'] == 'Positif']['Komentar Bersih'].astype(str))
        text_neg = " ".join(df[df['Label'] == 'Negatif']['Komentar Bersih'].astype(str))
        
        # Generate gambar
        img_pos = self._create_wc_image(text_pos, 'Greens')
        img_neg = self._create_wc_image(text_neg, 'Reds')
        
        return {'positive': img_pos, 'negative': img_neg}
    
        # TAMBAHKAN FUNGSI INI DI DALAM CLASS:
# --- UPDATE BAGIAN INI DI DALAM class SentimentModel (ml_engine.py) ---
    
    def generate_report_assets(self, filepath):
        """
        Membuat SEMUA grafik untuk laporan PDF:
        1. Pie Chart
        2. Confusion Matrix
        3. Wordcloud Positif & Negatif
        4. Bar Chart Perbandingan Model
        """
        if not self.is_trained:
            return None, "Model belum dilatih"

        df = pd.read_excel(filepath)
        X = df['Komentar Bersih'].astype(str)
        y = df['Label']
        
        # 1. Tentukan Model Terbaik
        best_name = max(self.results, key=lambda k: self.results[k]['accuracy'])
        best_model = self.models[best_name]
        
        # --- A. PIE CHART ---
        stats = {
            'total': len(df),
            'pos': len(df[df['Label'] == 'Positif']),
            'neg': len(df[df['Label'] == 'Negatif'])
        }
        
        plt.figure(figsize=(6, 6))
        plt.pie([stats['pos'], stats['neg']], labels=['Positif', 'Negatif'], 
                autopct='%1.1f%%', colors=['#22c55e', '#ef4444'], startangle=90, 
                textprops={'fontsize': 12})
        plt.title("Distribusi Sentimen Pasien", fontsize=14, fontweight='bold')
        
        img_pie = io.BytesIO()
        plt.savefig(img_pie, format='png', bbox_inches='tight')
        img_pie.seek(0)
        plt.close()

        # --- B. CONFUSION MATRIX (Model Terbaik) ---
        X_vec = self.vectorizer.transform(X)
        preds = best_model.predict(X_vec)
        cm = confusion_matrix(y, preds)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'],
                    annot_kws={"size": 14})
        plt.title(f"Akurasi Prediksi ({best_name})", fontsize=14, fontweight='bold')
        plt.ylabel('Aktual', fontsize=11)
        plt.xlabel('Prediksi AI', fontsize=11)
        
        img_cm = io.BytesIO()
        plt.savefig(img_cm, format='png', bbox_inches='tight')
        img_cm.seek(0)
        plt.close()

        # --- C. WORDCLOUDS (Positif & Negatif) ---
        text_pos = " ".join(df[df['Label'] == 'Positif']['Komentar Bersih'].astype(str))
        text_neg = " ".join(df[df['Label'] == 'Negatif']['Komentar Bersih'].astype(str))
        
        # WC Positif
        wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(text_pos if text_pos.strip() else "Data Kosong")
        img_wc_pos = io.BytesIO()
        wc_pos.to_image().save(img_wc_pos, format='PNG')
        img_wc_pos.seek(0)
        
        # WC Negatif
        wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text_neg if text_neg.strip() else "Data Kosong")
        img_wc_neg = io.BytesIO()
        wc_neg.to_image().save(img_wc_neg, format='PNG')
        img_wc_neg.seek(0)

        # --- D. BAR CHART PERBANDINGAN MODEL ---
        model_names = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in model_names]
        
        plt.figure(figsize=(8, 4))
        bars = plt.bar(model_names, accuracies, color=['#3b82f6', '#10b981', '#f59e0b'])
        plt.ylim(0, 100)
        plt.title("Perbandingan Akurasi Model AI", fontsize=14, fontweight='bold')
        plt.ylabel("Akurasi (%)")
        
        # Tambah label angka di atas bar
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval}%", ha='center', va='bottom', fontweight='bold')

        img_bar = io.BytesIO()
        plt.savefig(img_bar, format='png', bbox_inches='tight')
        img_bar.seek(0)
        plt.close()

        return {
            'stats': stats,
            'metrics': self.results[best_name], # Metrik model terbaik
            'all_models': self.results,         # Metrik semua model
            'best_model': best_name,
            'pie_bytes': img_pie.getvalue(),
            'cm_bytes': img_cm.getvalue(),
            'wc_pos_bytes': img_wc_pos.getvalue(),
            'wc_neg_bytes': img_wc_neg.getvalue(),
            'bar_bytes': img_bar.getvalue()
        }, None
# Instance Global
ai_engine = SentimentModel()