import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# Load and preprocess dataset
def load_data():
    df = pd.read_csv('movies.csv')
    df['genres'] = df['genres'].fillna('').str.replace('|', ' ')
    return df

# Create similarity matrix
def create_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Recommend function
def recommend_movie(title, df, cosine_sim, indices, num_recommendations=5):
    if title not in indices:
        return []
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices].tolist()

# GUI Functionality
def on_recommend():
    user_input = entry.get()
    if not user_input.strip():
        messagebox.showwarning("Input Error", "Please enter a movie title.")
        return

    recs = recommend_movie(user_input, df, cosine_sim, indices)
    if not recs:
        messagebox.showinfo("No Match", "Movie not found in dataset.")
    else:
        output.delete(0, tk.END)
        for rec in recs:
            output.insert(tk.END, rec)

# Load data
df = load_data()
cosine_sim = create_similarity(df)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

# Set up GUI
root = tk.Tk()
root.title("Movie Recommender")

tk.Label(root, text="Enter Movie Title:").pack(pady=5)

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

btn = tk.Button(root, text="Recommend", command=on_recommend)
btn.pack(pady=5)

tk.Label(root, text="Recommended Movies:").pack(pady=5)

output = tk.Listbox(root, width=50, height=6)
output.pack(pady=5)

root.mainloop()
