import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt  # Tambahan untuk grafik
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model dan vectorizer
model = joblib.load("model_naive_bayes.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Judul halaman
st.title("📰 Deteksi Berita Hoaks dengan Naive Bayes")
st.write("Masukkan teks berita untuk mengetahui apakah termasuk **Hoaks** atau **Fakta**.")

# Input berita dari pengguna
input_berita = st.text_area("📝 Masukkan isi atau judul berita:")

# Tombol Prediksi
if st.button("🔍 Prediksi"):
    if input_berita.strip() != "":
        # Vektorisasi
        vektor = vectorizer.transform([input_berita])
        prediksi = model.predict(vektor)[0]
        probabilitas = model.predict_proba(vektor)[0]  # [Fakta, Hoaks]

        # Tampilkan hasil teks
        st.subheader("📊 Hasil Prediksi:")
        if prediksi == 0:
            st.success(f"✅ Berita ini diprediksi sebagai **FAKTA** ({probabilitas[0]*100:.2f}%)")
        else:
            st.error(f"❌ Berita ini diprediksi sebagai **HOAKS** ({probabilitas[1]*100:.2f}%)")

        # Tampilkan grafik probabilitas
        st.markdown("### 📈 Probabilitas Prediksi")
        fig, ax = plt.subplots()
        kelas = ['Fakta', 'Hoaks']
        warna = ['green', 'red']
        ax.barh(kelas, probabilitas, color=warna)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas")
        ax.set_title("Visualisasi Probabilitas Prediksi")
        for i, v in enumerate(probabilitas):
            ax.text(v + 0.01, i, f"{v*100:.2f}%", color='white', fontweight='bold')
        st.pyplot(fig)
    else:
        st.warning("Masukkan teks berita terlebih dahulu.")
