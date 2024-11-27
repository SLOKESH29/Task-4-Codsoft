import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample dataset: Movies and their genres
data = {
    "Movie": ["The Matrix", "John Wick", "Inception", "Toy Story", "Finding Nemo", "Shrek"],
    "Genre": ["Sci-Fi Action", "Action Thriller", "Sci-Fi Thriller", "Animation Adventure", "Animation Adventure", "Animation Comedy"]
}

# Create a DataFrame
movies_df = pd.DataFrame(data)

# Display the dataset
print("Movies Dataset:")
print(movies_df)

# Step 1: Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['Genre'])

# Step 2: Calculate similarity matrix
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# Recommendation function
def recommend_movies(movie_name, num_recommendations=3):
    if movie_name not in movies_df['Movie'].values:
        return f"'{movie_name}' not found in the dataset."
    
    # Get index of the movie
    movie_idx = movies_df[movies_df['Movie'] == movie_name].index[0]
    
    # Get pairwise similarity scores
    similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
    
    # Sort by similarity scores (descending)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Exclude the first one (itself) and get top recommendations
    top_recommendations = similarity_scores[1:num_recommendations + 1]
    
    # Fetch movie names
    recommended_movies = [movies_df.iloc[idx]['Movie'] for idx, _ in top_recommendations]
    return recommended_movies

# Example usage
movie_to_search = "Inception"
print(f"\nRecommendations for '{movie_to_search}':")
print(recommend_movies(movie_to_search))
