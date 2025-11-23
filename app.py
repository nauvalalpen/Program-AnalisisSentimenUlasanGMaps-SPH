import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from streamlit_option_menu import option_menu
import plotly.express as px

st.set_page_config(
    page_title="Analisis Sentimen RS Semen Padang",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
    h1, h2, h3 {
        color: #d32f2f; /* Merah Semen Padang */
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("LogoSPH.png", width=200)
    st.title("Navigasi Riset")
    
    selected = option_menu(
        menu_title=None,
        options=["Dataset & Distribusi", "Wordcloud", "Model Naive Bayes", "Evaluasi & Hasil", "Prediksi Real-time"],
        icons=["table", "cloud", "cpu", "graph-up", "search"],
        menu_icon="cast",
        default_index=0,
        styles={
            "nav-link-selected": {"background-color": "#d32f2f"},
        }
    )
    st.info("Aplikasi ini disusun berdasarkan metodologi Naive Bayes Classifier.")

#Load Data
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)
    return df

#Training Model
def train_model(df):
    X = df['Komentar Bersih'].astype(str)
    y = df['Label']
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(X)
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
    
    # Naive Bayes Classifier
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    return model, tfidf, y_test, y_pred, X_test

#Dataset
if selected == "Dataset & Distribusi":
    st.title("📂 Eksplorasi Data Ulasan")
    st.markdown("Halaman ini menampilkan data bersih hasil *preprocessing* dan distribusi sentimen.")
    
    uploaded_file = st.file_uploader("Upload file Excel (Dataset Bersih)", type=['xlsx'])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        st.session_state['df'] = df
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Tabel Data Bersih")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total Data: {df.shape[0]} baris.")
            
        with col2:
            st.subheader("Distribusi Sentimen")
            count_data = df['Label'].value_counts().reset_index()
            count_data.columns = ['Label', 'Jumlah']
            fig = px.pie(count_data, values='Jumlah', names='Label', 
                         color='Label', color_discrete_map={'Positif':'#66b3ff', 'Negatif':'#ff9999'},
                         title='Persentase Sentimen')
            st.plotly_chart(fig, use_container_width=True)

#Wordcloud
elif selected == "Wordcloud":
    st.title("☁️ Visualisasi Wordcloud")
    st.markdown("Visualisasi kata yang paling sering muncul pada ulasan Positif dan Negatif (Sesuai **Jurnal Hal 736, Gambar 9 & 10**).")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        col1, col2 = st.columns(2)
        
        # Wordcloud Positif
        with col1:
            st.subheader("Wordcloud Sentimen Positif")
            text_pos = " ".join(df[df['Label'] == 'Positif']['Komentar Bersih'].astype(str))
            if text_pos:
                wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(text_pos)
                fig, ax = plt.subplots()
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Tidak ada data positif.")

        # Wordcloud Negatif
        with col2:
            st.subheader("Wordcloud Sentimen Negatif")
            text_neg = " ".join(df[df['Label'] == 'Negatif']['Komentar Bersih'].astype(str))
            if text_neg:
                wc_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text_neg)
                fig, ax = plt.subplots()
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Tidak ada data negatif.")
    else:
        st.error("Silakan upload data terlebih dahulu di menu 'Dataset'.")

#Model Naive Bayes
elif selected == "Model Naive Bayes":
    st.title("⚙️ Pemodelan Naive Bayes")
    st.markdown("Proses transformasi TF-IDF dan Pelatihan Model (Sesuai **Jurnal Hal 735**).")
    
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        if st.button("Mulai Training Model"):
            with st.spinner("Sedang melatih model..."):
                model, tfidf, y_test, y_pred, X_test = train_model(df)
                
                st.session_state['model'] = model
                st.session_state['tfidf'] = tfidf
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                
            st.success("Model berhasil dilatih!")
            
            st.markdown("### Detail Parameter:")
            st.write("- **Metode:** Multinomial Naive Bayes")
            st.write("- **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency)")
            st.write("- **Data Split:** 70% Training, 30% Testing")
    else:
        st.error("Silakan upload data terlebih dahulu.")

#Evaluasi
elif selected == "Evaluasi & Hasil":
    st.title("📊 Evaluasi Kinerja Model")
    st.markdown("Menampilkan Confusion Matrix dan Metrik Akurasi (Sesuai **Jurnal Hal 736, Tabel 9 & Gambar 8**).")
    
    if 'y_test' in st.session_state:
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label='Positif', average='weighted')
        rec = recall_score(y_test, y_pred, pos_label='Positif', average='weighted')
        f1 = f1_score(y_test, y_pred, pos_label='Positif', average='weighted')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc*100:.2f}%")
        col2.metric("Precision", f"{prec*100:.2f}%")
        col3.metric("Recall", f"{rec*100:.2f}%")
        col4.metric("F1-Score", f"{f1*100:.2f}%")
        
        st.divider()

        col_cm, col_rep = st.columns([1, 2])
        
        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            labels = sorted(list(set(y_test)))
            
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig_cm)
            
        with col_rep:
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.dataframe(df_report.style.highlight_max(axis=0), use_container_width=True)
            
    else:
        st.warning("Model belum dilatih. Silakan pergi ke menu 'Model Naive Bayes' dan klik tombol Training.")

#Prediksi
elif selected == "Prediksi Real-time":
    st.title("🤖 Uji Coba Prediksi")
    st.markdown("Uji model yang telah dilatih dengan memasukkan ulasan baru.")
    
    if 'model' in st.session_state:
        model = st.session_state['model']
        tfidf = st.session_state['tfidf']
        
        input_text = st.text_area("Masukkan ulasan tentang RS Semen Padang:", placeholder="Contoh: Pelayanan sangat ramah dan cepat.")
        
        if st.button("Analisis Sentimen"):
            if input_text:
                text_vector = tfidf.transform([input_text])
                prediction = model.predict(text_vector)[0]
                proba = model.predict_proba(text_vector)
                
                st.divider()
                if prediction == "Positif":
                    st.success(f"**Hasil: POSITIF**")
                else:
                    st.error(f"**Hasil: NEGATIF**")
                
                st.write(f"Confidence Score: {np.max(proba)*100:.2f}%")
            else:
                st.warning("Mohon masukkan teks ulasan.")
    else:
        st.error("Model belum siap. Lakukan training data terlebih dahulu.")