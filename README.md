# 🎬 Movie Recommendation System

## About the Project

This project recommends movies with similar genres based on the movie selected by the user. It is built using Python and machine learning techniques.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit

## Features

* Select a movie from a dropdown menu.
* Choose the number of recommendations.
* Find movies with similar genres.
* Display movie genres and sample ratings.
* Calculate genre similarity percentages.

## Machine Learning

The project uses CountVectorizer to convert movie genres into numerical features and cosine similarity to measure similarity between movies.

## Installation

Install the required libraries:

```
python -m pip install pandas scikit-learn streamlit
```

## Run the Application

Execute this command in the terminal:

```
python -m streamlit run app.py
```

## Note

This project uses a small sample dataset and demonstration ratings. Similarity percentages represent genre similarity, not actual audience preferences.
  
