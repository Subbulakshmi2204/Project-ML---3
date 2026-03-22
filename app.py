import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System with User Clustering")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

df = load_data()

st.subheader("📊 Movie Dataset")
st.dataframe(df)

# -------------------------------
# Preprocessing
# -------------------------------
le = LabelEncoder()
df['genre_encoded'] = le.fit_transform(df['genre'])

# -------------------------------
# K-Means Clustering
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['rating', 'popularity', 'genre_encoded']])

# -------------------------------
# Cluster Meaning (Interpretation)
# -------------------------------
cluster_meaning = {
    0: "🔥 Action & Popular Movie Lover",
    1: "❤️ Romance & Emotional Movie Lover",
    2: "🎯 Casual Viewer (Explores Mixed Content)"
}

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🎯 Enter Your Preferences")

user_rating = st.sidebar.slider("Preferred Rating", 1.0, 5.0, 4.5)
user_popularity = st.sidebar.slider("Popularity Preference", 50, 100, 90)
user_genre = st.sidebar.selectbox("Preferred Genre", df['genre'].unique())

user_genre_encoded = le.transform([user_genre])[0]

# -------------------------------
# Predict User Cluster
# -------------------------------
user_cluster = kmeans.predict([[user_rating, user_popularity, user_genre_encoded]])[0]

# -------------------------------
# Show User Profile
# -------------------------------
st.subheader("🧠 Your Movie Personality")
st.success(cluster_meaning[user_cluster])

# -------------------------------
# Recommendations
# -------------------------------
recommended = df[df['cluster'] == user_cluster]

top_movies = recommended.sort_values(by='rating', ascending=False).head(5)

st.subheader("🏆 Top Recommended Movies For You")

for i, row in top_movies.iterrows():
    st.write(f"🎬 {row['title']} ({row['genre']}) ⭐ {row['rating']}")

# -------------------------------
# Why These Recommendations
# -------------------------------
st.subheader("📌 Why These Movies?")

st.info(f"""
You selected:
- Preferred Rating: {user_rating}
- Popularity Preference: {user_popularity}
- Genre: {user_genre}

👉 These movies are recommended because they match users with similar preferences (same cluster).
""")

# -------------------------------
# Insights Section
# -------------------------------
st.subheader("📊 Insights About You")

if user_rating > 4.5:
    st.write("✔ You prefer highly rated movies")
else:
    st.write("✔ You are open to average-rated movies")

if user_popularity > 85:
    st.write("✔ You like trending/popular movies")
else:
    st.write("✔ You enjoy less mainstream content")

# -------------------------------
# Random Forest (Optional Prediction)
# -------------------------------
st.subheader("⭐ Predicted Rating (ML Model)")

rf = RandomForestRegressor()
rf.fit(df[['popularity', 'genre_encoded']], df['rating'])

pred_rating = rf.predict([[user_popularity, user_genre_encoded]])[0]

st.write(f"👉 Estimated rating for your taste: ⭐ {round(pred_rating, 2)}")

# -------------------------------
# Visualization
# -------------------------------
st.subheader("📈 Clustering Visualization")

fig, ax = plt.subplots()

for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    ax.scatter(cluster_data['rating'], cluster_data['popularity'], label=f'Cluster {cluster}')

# Plot user point
ax.scatter(user_rating, user_popularity, marker='X', s=200, label='You')

ax.set_xlabel("Rating")
ax.set_ylabel("Popularity")
ax.legend()

st.pyplot(fig)

# -------------------------------
# Footer Insight
# -------------------------------
st.subheader("📢 Final Conclusion")

st.success("""
✔ Users are grouped into clusters based on preferences  
✔ Similar users like similar types of movies  
✔ This improves personalized recommendations  

👉 Clustering helps discover hidden patterns in user behavior.
""")
