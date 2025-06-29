import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Baca dataset
df =df = pd.read_csv(r"C:\Users\silvi\Documents\SEMESTER 6\DATA MINING\New folder\berita_hoax.csv")
  # pastikan file CSV ini ada di folder yang sama

# Pisahkan fitur dan label
X = df['text']
y = df['label']

# TF-IDF vektorisasi teks
vectorizer = TfidfVectorizer ()

X_vector = vectorizer.fit_transform(X)

# Split data untuk training
X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

# Latih model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Simpan model dan vectorizer
joblib.dump(model, "model_naive_bayes.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model dan vectorizer berhasil disimpan!")
