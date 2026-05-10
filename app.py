import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# ======================
# LOAD DATA
# ======================
try:
    df = pd.read_csv("data/data_film.csv")
    df.columns = df.columns.str.strip().str.lower()
except Exception as e:
    print(f"Error loading CSV: {e}")
    df = pd.DataFrame(columns=['judul', 'sinopsis', 'genre', 'tahun', 'rate', 'kata_kunci'])

# ======================
# CLEAN DATA TYPES
# ======================
df['tahun'] = pd.to_numeric(df['tahun'], errors='coerce').fillna(0).astype(int)
df['rate'] = pd.to_numeric(df['rate'], errors='coerce').fillna(0) # Default ke 0 sesuai request UI

df['sinopsis'] = df['sinopsis'].fillna('')
df['genre'] = df['genre'].fillna('')
df['kata_kunci'] = df['kata_kunci'].fillna('')

# ======================
# CLEAN TEXT & STOPWORDS
# ======================
stop_words_id = [
    'yang', 'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dengan', 'dan', 'atau',
    'ini', 'itu', 'juga', 'sudah', 'saya', 'dia', 'mereka', 'sebuah', 'adalah'
]

def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['sinopsis_clean'] = df['sinopsis'].apply(clean)
df['kata_kunci_clean'] = df['kata_kunci'].apply(clean)
df['genre_clean'] = df['genre'].apply(clean)

# ======================
# FEATURE ENGINEERING
# ======================
df['combined_text'] = (
    (df['sinopsis_clean'] + " ") * 8 +
    (df['kata_kunci_clean'] + " ") * 4 +
    (df['genre_clean'] + " ") * 2
)

# ======================
# TF-IDF VECTORIZER
# ======================
tfidf = TfidfVectorizer(
    stop_words=stop_words_id,
    ngram_range=(1, 2),
    max_features=30000,
    sublinear_tf=True
)

if not df.empty:
    tfidf_matrix = tfidf.fit_transform(df['combined_text'])
else:
    tfidf_matrix = None

# ======================
# RECOMMENDATION ENGINE
# ======================
def recommend(text, input_genre, input_year):
    if tfidf_matrix is None or not text.strip():
        return []

    text = clean(text)
    vec = tfidf.transform([text])
    sim = cosine_similarity(vec, tfidf_matrix).flatten()

    temp = df.copy()
    temp['similarity'] = sim

    # Ambil yang ada kemiripan sinopsisnya (meskipun kecil)
    temp = temp[temp['similarity'] > 0]
    if temp.empty:
        return []

    # 1. SCORING GENRE (10%)
    if input_genre and input_genre.lower().strip() not in ["", "semua genre"]:
        # Pisahkan inputan user berdasarkan koma, contoh: "action, drama" -> ["action", "drama"]
        user_genres = [clean(g) for g in input_genre.split(',') if g.strip()]
        
        def calc_genre_score(movie_genre_str):
            if not user_genres: return 1.0
            mg = str(movie_genre_str).lower()
            # Hitung berapa banyak genre inputan user yang ada di film ini
            matched = sum(1 for ug in user_genres if ug in mg)
            return matched / len(user_genres) # Skala 0.0 - 1.0
            
        temp['genre_score'] = temp['genre'].apply(calc_genre_score)
    else:
        temp['genre_score'] = 1.0 # Jika tidak diisi / Semua Genre, maka skor penuh

    # 2. SCORING TAHUN (5%)
    if input_year and str(input_year).lower().strip() != "semua tahun":
        target_year = int(input_year)
        # Soft scoring: Skala mengecil jika tahun makin jauh bedanya (Exact match = 1.0)
        temp['year_score'] = 1.0 / (1.0 + (temp['tahun'] - target_year).abs())
    else:
        temp['year_score'] = 1.0

    # 3. FINAL SCORE (85% Sinopsis + 10% Genre + 5% Tahun)
    temp['final_score'] = (
        0.85 * temp['similarity'] +
        0.10 * temp['genre_score'] +
        0.05 * temp['year_score']
    )

    # Sort berdasarkan skor akhir
    result = temp.sort_values('final_score', ascending=False)
    
    return result[['judul', 'tahun', 'genre', 'rate', 'final_score', 'similarity', 'sinopsis']].to_dict('records')

# ======================
# ROUTES
# ======================

@app.route('/stats')
def stats():
    all_genres = df['genre'].str.split(',').explode().str.strip().dropna()
    all_genres = all_genres[all_genres != ""]
    genre_counts = all_genres.value_counts()
    
    valid_years = df[df['tahun'] > 0]
    year_counts = valid_years['tahun'].value_counts().sort_index(ascending=False).head(10)
    last_10_years = year_counts.sort_index(ascending=True).to_dict()
    last_10_years_clean = {str(k): int(v) for k, v in last_10_years.items()}

    data = {
        "total_film": int(len(df)),
        "total_genre": int(all_genres.nunique()),
        "top_10_genre": genre_counts.head(10).to_dict(), 
        "last_10_years": last_10_years_clean 
    }
    return jsonify(data)

@app.route('/')
def index():
    years = sorted(df['tahun'].unique(), reverse=True)
    # Tidak lagi mengirim 'genres' karena UI sudah diganti input text biasa
    return render_template('index.html', years=years)

@app.route('/recommend', methods=['POST'])
def rec():
    data = request.get_json()
    synopsis = data.get('synopsis', '')
    genre = data.get('genre', '')
    year = data.get('year', 'Semua Tahun')

    if not synopsis.strip():
        return jsonify({'error': 'Sinopsis kosong'}), 400

    result = recommend(synopsis, genre, year)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)


    