import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🎬 Movie Recommendation System with User Clustering")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

df = load_data()

st.subheader("📊 Dataset")
st.write(df)

# Encode genre
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['rating', 'popularity', 'genre_encoded']])

# User Input
st.sidebar.header("🎯 Enter Your Preferences")

user_rating = st.sidebar.slider("Preferred Rating", 1.0, 5.0, 4.5)
user_popularity = st.sidebar.slider("Popularity Preference", 50, 100, 90)
user_genre = st.sidebar.selectbox("Genre", df['genre'].unique())

user_genre_encoded = le.transform([user_genre])[0]

# Predict cluster for user
user_cluster = kmeans.predict([[user_rating, user_popularity, user_genre_encoded]])[0]

st.subheader("🧠 Your Cluster Group")
st.write(f"You belong to cluster: {user_cluster}")

# Recommend movies from same cluster
recommended = df[df['cluster'] == user_cluster]

st.subheader("🎥 Recommended Movies")
st.write(recommended[['title', 'genre', 'rating']])

# Visualization
st.subheader("📈 Cluster Visualization")
st.scatter_chart(df[['rating', 'popularity']])
