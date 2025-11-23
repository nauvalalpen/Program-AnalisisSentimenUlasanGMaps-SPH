import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import plotly.express as px
import plotly.graph_objects as go

# Import modul buatan sendiri
import database
import utils

# ==========================================
# 1. KONFIGURASI HALAMAN & STATE
# ==========================================
st.set_page_config(
    page_title="SPH Sentiment AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi Database
database.init_db()

# Custom CSS untuk UI Modern
st.markdown("""
<style>
    .stApp {background-color: #f8f9fa;}
    .main-header {font-size: 2.5rem; color: #d32f2f; font-weight: 700;}
    .sub-header {font-size: 1.2rem; color: #555;}
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    [data-testid="stMetricValue"] {font-size: 24px; color: #d32f2f;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. FUNGSI LOGIKA (BACKEND)
# ==========================================

@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

def train_comparative_models(df):
    """Melatih 3 Model sekaligus untuk Perbandingan"""
    X = df['Komentar Bersih'].astype(str)
    y = df['Label']
    
    tfidf = TfidfVectorizer(max_features=2000)
    X_tfidf = tfidf.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear)": SVC(kernel='linear', probability=True)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label='Positif', average='weighted')
        results[name] = {"Accuracy": acc, "F1-Score": f1, "y_pred": y_pred}
        trained_models[name] = model
        
    return results, trained_models, tfidf, X_test, y_test

# ==========================================
# 3. SIDEBAR & NAVIGASI
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/83/Logo_Semen_Padang_Hospital.png/320px-Logo_Semen_Padang_Hospital.png", width=180)
    st.markdown("---")
    
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Dataset & Aspek", "Komparasi Model", "Evaluasi & Laporan", "Prediksi & History"],
        icons=["house", "table", "bar-chart-steps", "file-earmark-pdf", "clock-history"],
        menu_icon="list",
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#d32f2f"}}
    )
    
    st.markdown("---")
    st.caption("© 2025 Riset Semen Padang Hospital")

# ==========================================
# 4. KONTEN HALAMAN UTAMA
# ==========================================

# --- MENU: HOME (LANDING PAGE) ---
if selected == "Home":
    col_hero, col_anim = st.columns([1.5, 1])
    
    with col_hero:
        st.markdown('<div class="main-header">Sistem Analisis Sentimen Cerdas</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Membantu Manajemen Rumah Sakit Semen Padang memahami kepuasan pasien melalui ulasan Google Maps menggunakan Kecerdasan Buatan.</div>', unsafe_allow_html=True)
        st.write("")
        st.write("🚀 **Fitur Unggulan:**")
        st.write("✅ **Multi-Model AI:** Membandingkan Naive Bayes, SVM, & Logistic Regression.")
        st.write("✅ **Aspect Filtering:** Analisis spesifik (Dokter, Antrian, Parkir).")
        st.write("✅ **PDF Reporting:** Download laporan otomatis untuk rapat direksi.")
        st.write("✅ **Database History:** Menyimpan riwayat prediksi.")
        
        st.info("Silakan pilih menu di sidebar untuk memulai analisis.")

    with col_anim:
        # Load animasi robot/dokter
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_5njp3vgg.json" 
        lottie_json = utils.load_lottieurl(lottie_url)
        st_lottie(lottie_json, height=400)

# --- MENU: DATASET & ASPEK (ASPECT BASED) ---
elif selected == "Dataset & Aspek":
    st.title("📂 Dataset & Filter Aspek")
    
    uploaded_file = st.file_uploader("Upload Data Excel (Bersih)", type=['xlsx'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state['df'] = df
        
        # --- FITUR INOVASI: FILTER ASPEK ---
        st.markdown("### 🔍 Filter Berdasarkan Topik (Aspect-Based)")
        keyword = st.text_input("Cari kata kunci spesifik (misal: 'dokter', 'antrian', 'obat'):", "")
        
        if keyword:
            filtered_df = df[df['Komentar Bersih'].str.contains(keyword, case=False, na=False)]
            st.success(f"Ditemukan {len(filtered_df)} ulasan mengandung kata '{keyword}'.")
        else:
            filtered_df = df
            
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(filtered_df.head(10), use_container_width=True)
        with col2:
            # Distribusi Sentimen Interaktif
            fig = px.pie(filtered_df, names='Label', title=f"Sentimen untuk: {keyword if keyword else 'Semua Data'}",
                         color='Label', color_discrete_map={'Positif':'#2ecc71', 'Negatif':'#e74c3c'})
            st.plotly_chart(fig, use_container_width=True)
            
        # Wordcloud untuk Data Terfilter
        st.subheader(f"Apa kata mereka tentang '{keyword if keyword else 'Semua'}'?")
        text = " ".join(filtered_df['Komentar Bersih'].astype(str))
        if text:
            wc = WordCloud(width=800, height=300, background_color='white').generate(text)
            fig_wc, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
    else:
        st.warning("Upload data dataset_bersih.xlsx terlebih dahulu.")

# --- MENU: KOMPARASI MODEL (BENCHMARKING) ---
elif selected == "Komparasi Model":
    st.title("⚖️ Perbandingan Algoritma AI")
    st.markdown("Membandingkan performa Naive Bayes dengan model lain untuk membuktikan validitas metode.")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        if st.button("Latih & Bandingkan Semua Model"):
            with st.spinner("Melatih Naive Bayes, SVM, dan Logistic Regression..."):
                results, trained_models, tfidf, X_test, y_test = train_comparative_models(df)
                
                # Simpan ke session state
                st.session_state['results'] = results
                st.session_state['trained_models'] = trained_models
                st.session_state['tfidf'] = tfidf
                st.session_state['y_test'] = y_test
                st.session_state['X_test'] = X_test # Perlu untuk report
                
            # Visualisasi Perbandingan Akurasi
            res_df = pd.DataFrame(results).T.reset_index()
            res_df.rename(columns={'index': 'Model'}, inplace=True)
            
            st.subheader("Hasil Akurasi Model")
            fig_bar = px.bar(res_df, x='Model', y='Accuracy', color='Model', text_auto='.2%',
                             title="Perbandingan Tingkat Akurasi")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Highlight Pemenang
            best_model = res_df.loc[res_df['Accuracy'].idxmax()]
            st.success(f"🏆 Model Terbaik: **{best_model['Model']}** dengan Akurasi {best_model['Accuracy']*100:.2f}%")
            
    else:
        st.error("Silakan upload data di menu Dataset.")

# --- MENU: EVALUASI & LAPORAN (PDF REPORT) ---
elif selected == "Evaluasi & Laporan":
    st.title("📊 Evaluasi Mendalam & Laporan")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        model_name = st.selectbox("Pilih Model untuk Dievaluasi Detail:", list(results.keys()))
        
        y_test = st.session_state['y_test']
        y_pred = results[model_name]['y_pred']
        
        # 1. Confusion Matrix
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
            st.pyplot(fig_cm)
            
        with col2:
            st.subheader("Metrik Performa")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).T.style.highlight_max(axis=0))

        # --- FITUR INOVASI: PDF DOWNLOAD ---
        st.markdown("---")
        st.subheader("📄 Generate Laporan Manajerial")
        st.markdown("Unduh laporan otomatis siap cetak untuk presentasi manajemen.")
        
        if st.button("Buat Laporan PDF"):
            # Siapkan Data untuk PDF
            df = st.session_state['df']
            counts = df['Label'].value_counts(normalize=True) * 100
            pos_pct = counts.get('Positif', 0)
            neg_pct = counts.get('Negatif', 0)
            acc = results[model_name]['Accuracy'] * 100
            f1 = results[model_name]['F1-Score'] * 100
            
            # Buat Grafik Pie khusus untuk PDF (Plotly static)
            fig_pie_static = px.pie(values=[pos_pct, neg_pct], names=['Positif', 'Negatif'])
            
            # Generate
            pdf_bytes = utils.generate_pdf(len(df), pos_pct, neg_pct, acc, f1, fig_pie_static, fig_cm)
            
            st.download_button(
                label="📥 Download Laporan PDF",
                data=pdf_bytes,
                file_name="Laporan_Analisis_Sentimen_SPH.pdf",
                mime="application/pdf"
            )
            
    else:
        st.warning("Latih model terlebih dahulu di menu Komparasi Model.")

# --- MENU: PREDIKSI & HISTORY (DATABASE) ---
elif selected == "Prediksi & History":
    st.title("🤖 Prediksi & Database")
    
    if 'trained_models' in st.session_state:
        models = st.session_state['trained_models']
        tfidf = st.session_state['tfidf']
        
        # Tab Navigasi dalam Halaman
        tab1, tab2 = st.tabs(["Uji Coba Prediksi", "Riwayat Database"])
        
        with tab1:
            col_input, col_res = st.columns([2, 1])
            with col_input:
                input_text = st.text_area("Masukkan ulasan baru:", height=100)
                active_model = st.selectbox("Pilih Model Prediksi:", list(models.keys()))
                
            with col_res:
                st.write("") # Spacer
                st.write("") 
                if st.button("Analisis Sekarang", type="primary"):
                    if input_text:
                        # Proses Prediksi
                        vec = tfidf.transform([input_text])
                        model = models[active_model]
                        pred = model.predict(vec)[0]
                        proba = np.max(model.predict_proba(vec)) * 100
                        
                        # Tampilkan Hasil
                        if pred == "Positif":
                            st.success(f"**POSITIF** ({proba:.2f}%)")
                        else:
                            st.error(f"**NEGATIF** ({proba:.2f}%)")
                            
                        # SIMPAN KE DATABASE
                        database.save_prediction(input_text, pred, proba, active_model)
                        st.toast("Data berhasil disimpan ke Database!", icon="💾")
                    else:
                        st.warning("Input teks kosong!")
                        
        with tab2:
            st.subheader("📚 Riwayat Prediksi (SQLite)")
            if st.button("Refresh Data"):
                st.rerun()
                
            history_df = database.load_history()
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("Hapus Riwayat"):
                database.clear_history()
                st.rerun()
                
    else:
        st.error("Model belum siap. Lakukan training dahulu.")