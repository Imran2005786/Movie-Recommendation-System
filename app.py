import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
page_title="Movie Recommendation System",
page_icon="🎬",
layout="centered"
)

st.title("🎬 Movie Recommendation System")
st.write("Discover movies similar to your favourites!")

movies = pd.DataFrame({
"title": [
"Avatar", "Titanic", "The Avengers", "Iron Man",
"Interstellar", "Inception", "The Dark Knight",
"Toy Story", "Finding Nemo", "Jurassic Park",
"Spider-Man", "The Lion King", "Gravity",
"Doctor Strange", "Jumanji"
],
"genre": [
"Action Adventure Sci-Fi",
"Romance Drama",
"Action Adventure Superhero",
"Action Sci-Fi Superhero",
"Sci-Fi Adventure Drama",
"Sci-Fi Action Thriller",
"Action Crime Thriller",
"Animation Comedy Family",
"Animation Adventure Family",
"Adventure Sci-Fi Thriller",
"Action Adventure Superhero",
"Animation Adventure Drama",
"Sci-Fi Thriller Drama",
"Action Fantasy Superhero",
"Adventure Comedy Fantasy"
],
"rating": [
7.9, 7.9, 8.0, 7.9, 8.7,
8.8, 9.0, 8.3, 8.2, 8.2,
7.4, 8.5, 7.7, 7.5, 7.1
]
})


vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genre"])
similarity_matrix = cosine_similarity(genre_matrix)

selected_movie = st.selectbox(
"🎥 Choose a movie:",
movies["title"].tolist()
)

number_of_movies = st.slider(
"Number of recommendations:",
min_value=1,
max_value=5,
value=5
)
if st.button("Recommend Movies", type="primary"):
    movie_index = movies.index[
movies["title"] == selected_movie
][0]


scores = list(enumerate(similarity_matrix[movie_index]))
scores.sort(key=lambda item: item[1], reverse=True)

recommendations = [
    (index, score)
    for index, score in scores
    if index != movie_index and score > 0
]

st.subheader("🍿 Recommended Movies")

if recommendations:
    for index, score in recommendations[:number_of_movies]:
        movie_title = movies.iloc[index]["title"]
        movie_genre = movies.iloc[index]["genre"]
        movie_rating = movies.iloc[index]["rating"]

        st.success( 
            f"🎬 {movie_title}\n\n"
            f"Genre: {movie_genre}\n\n"
            f"Rating: {movie_rating}/10\n\n"
            f"Genre Similarity: {score:.0%}"
        )
        
else:
    st.info(
        "No similar movies found. Please choose another movie."
    )