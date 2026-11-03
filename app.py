
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Movie dataset
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

# Machine Learning model
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies["genre"])
similarity_matrix = cosine_similarity(genre_matrix)


# Home page
@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []
    selected_movie = movies["title"].iloc[0]
    number_of_movies = 5

    if request.method == "POST":

        selected_movie = request.form.get("movie", selected_movie)

        try:
            number_of_movies = int(request.form.get("count", 5))
            number_of_movies = max(1, min(5, number_of_movies))
        except ValueError:
            number_of_movies = 5

        if selected_movie in movies["title"].values:

            movie_index = movies.index[
                movies["title"] == selected_movie
            ][0]

            scores = list(
                enumerate(similarity_matrix[movie_index])
            )

            scores.sort(
                key=lambda item: item[1],
                reverse=True
            )

            for index, score in scores:

                if index != movie_index and score > 0:

                    recommendations.append({
                        "title": movies.iloc[index]["title"],
                        "genre": movies.iloc[index]["genre"],
                        "rating": movies.iloc[index]["rating"],
                        "similarity": f"{score:.0%}"
                    })

            recommendations = recommendations[:number_of_movies]

    return render_template(
        "index.html",
        movies=movies["title"].tolist(),
        selected_movie=selected_movie,
        number_of_movies=number_of_movies,
        recommendations=recommendations
    )


# Run Flask application
if __name__ == "__main__":
    app.run(debug=True)

