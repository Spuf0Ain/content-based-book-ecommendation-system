# recommender/content_recommender.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer # For handling list of genres
import numpy as np
import warnings
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
preprocessing_path = os.path.join(project_root, 'preprocessing')
if preprocessing_path not in sys.path:
     sys.path.append(preprocessing_path)

try:
    from text_cleaner import clean_text
    from feature_engineer import process_genres, normalize_numerical_features
except ImportError:
    warnings.warn("Could not import preprocessing modules. Ensure they are in the correct path.")
    def clean_text(text): return str(text) if isinstance(text, str) else ""
    def process_genres(df, **kwargs): return df 
    def normalize_numerical_features(df, **kwargs): return df 


class ContentRecommender:
    """
    A content-based recommendation system for books using TF-IDF on descriptions
    and incorporating genres and ratings.
    """
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the recommender with the dataset.

        Args:
            dataframe (pd.DataFrame): The DataFrame containing book data.
                                     Expected columns include 'Book', 'Description',
                                     'Genres', 'Avg_Rating', 'Num_Ratings'.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        self.df = dataframe.copy()
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self._prepare_data()
        self._build_model()

    def _prepare_data(self):
        """Prepares the data by cleaning text and processing features."""
        print("Preparing data...")
        # 1. Clean Descriptions
        if 'Description' in self.df.columns:
            print("Cleaning descriptions...")
            self.df['cleaned_description'] = self.df['Description'].apply(clean_text)
        else:
            warnings.warn("Column 'Description' not found. Skipping description cleaning.")
            self.df['cleaned_description'] = ''

        if 'Genres' in self.df.columns:
             print("Processing genres...")
             if isinstance(self.df['Genres'].iloc[0], str): # Basic check if processing needed
                 self.df = process_genres(self.df, genres_col='Genres')
             self.df['genres_str'] = self.df['Genres'].apply(lambda x: ' '.join(x) if isinstance(x, list) else '')
        else:
             warnings.warn("Column 'Genres' not found. Skipping genre processing.")
             self.df['genres_str'] = ''

        print("Normalizing ratings...")
        # Ensure normalization is applied (might be redundant)
        if 'Avg_Rating_normalized' not in self.df.columns:
            self.df = normalize_numerical_features(self.df, cols_to_normalize=['Avg_Rating', 'Num_Ratings'])

        # Create a combined feature for TF-IDF
        # Give more weight to genres maybe? Repeat genre string? IDK!
        # Example: self.df['combined_features'] = self.df['cleaned_description'] + ' ' + self.df['genres_str'] * 3
        self.df['combined_features'] = self.df['cleaned_description'] + ' ' + self.df['genres_str']

        self.df = self.df.reset_index(drop=True) # Ensure index is unique
        self.indices = pd.Series(self.df.index, index=self.df['Book']).drop_duplicates()
        print("Data preparation complete.")


    def _build_model(self):
        """Builds the TF-IDF matrix and computes cosine similarity."""
        print("Building TF-IDF model...")
        # Handle potential empty combined_features
        if 'combined_features' not in self.df.columns or self.df['combined_features'].isnull().all():
             warnings.warn("Column 'combined_features' is missing or empty. Cannot build TF-IDF model.")
             self.tfidf_matrix = None
             self.cosine_sim = None
             return

        # Initialize TF-IDF Vectorizer
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=3, stop_words='english')

        # Fill NaN with empty string to avoid errors
        self.tfidf_matrix = tfidf.fit_transform(self.df['combined_features'].fillna(''))
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        # Compute the cosine similarity matrix
        print("Calculating cosine similarity...")
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        print("Cosine similarity matrix calculated.")
        print("Model building complete.")

    def get_recommendations(self, book_title: str, k: int = 10, rating_weight: float = 0.1) -> pd.DataFrame:
        """
        Gets book recommendations based on content similarity and ratings.

        Args:
            book_title (str): The title of the book to get recommendations for.
            k (int): The number of recommendations to return.
            rating_weight (float): The weight given to normalized average rating
                                   when scoring recommendations (0 to 1).

        Returns:
            pd.DataFrame: A DataFrame containing the top k recommended books,
                          including their title, author, similarity score, and rating.
                          Returns empty DataFrame if book not found or model not built.
        """
        if self.cosine_sim is None or self.indices is None:
             print("Error: Model not built or data not prepared.")
             return pd.DataFrame()

        if book_title not in self.indices:
            # Try partial match or suggest alternatives?
            possible_matches = [title for title in self.indices.index if book_title.lower() in title.lower()]
            if not possible_matches:
                 print(f"Error: Book title '{book_title}' not found in the dataset.")
                 return pd.DataFrame()
            else:
                 actual_title = possible_matches[0]
                 print(f"Book title '{book_title}' not found. Using closest match: '{actual_title}'")
                 book_title = actual_title

        # Get the index of the book that matches the title
        idx = self.indices[book_title]

        # Get the pairwise similarity scores of all books with that book
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the books based on the similarity scores (descending)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the k most similar books (excluding the book itself)
        sim_scores = sim_scores[1:k+1]

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]
        content_similarity_values = [i[1] for i in sim_scores]

        # Get recommended book details
        recs_df = self.df.iloc[book_indices].copy()
        recs_df['similarity_score'] = content_similarity_values

        if 'Avg_Rating_normalized' in recs_df.columns and rating_weight > 0:
             recs_df['Avg_Rating_normalized'] = recs_df['Avg_Rating_normalized'].fillna(0)

             recs_df['combined_score'] = (
                 (1 - rating_weight) * recs_df['similarity_score'] +
                 rating_weight * recs_df['Avg_Rating_normalized']
             )
             # Sort by the combined score
             recs_df = recs_df.sort_values('combined_score', ascending=False)

        output_cols = ['Book', 'Author', 'Avg_Rating', 'Num_Ratings', 'similarity_score']
        if 'combined_score' in recs_df.columns:
             output_cols.append('combined_score')
        output_cols.append('Genres')
        output_cols.append('URL')      # Add URL

        return recs_df[output_cols]

if __name__ == '__main__':
    data = {
        'Book': ['The Great Gatsby', 'To Kill a Mockingbird', '1984', 'Pride and Prejudice', 'The Hobbit', 'SciFi Adventure', 'Another Novel'],
        'Author': ['F. Scott Fitzgerald', 'Harper Lee', 'George Orwell', 'Jane Austen', 'J.R.R. Tolkien','Author SciFi','Author Novel'],
        'Description': [
            'A story about the decadent elite of the Jazz Age.',
            'A novel about childhood innocence and prejudice in the American South.',
            'A dystopian novel about surveillance and totalitarianism.',
            'A romantic novel of manners.',
            'A fantasy adventure about hobbits, dwarves, and a dragon.',
            'An adventure in space with aliens and robots.',
            'A contemporary story about relationships.'
        ],
        'Genres': [
            "['Classic', 'Fiction']",
            "['Classic', 'Fiction']",
            "['Classic', 'Fiction', 'Sci-Fi', 'Dystopian']",
            "['Classic', 'Fiction', 'Romance']",
            "['Fantasy', 'Fiction', 'Adventure']",
            "['Sci-Fi', 'Adventure', 'Fiction']",
            "['Fiction', 'Contemporary']"
        ],
        'Avg_Rating': [3.9, 4.3, 4.2, 4.1, 4.3, 3.5, 4.0],
        'Num_Ratings': [1000, 2000, 1500, 1800, 2500, 500, 800],
        'URL': ['url_a', 'url_b', 'url_c', 'url_d', 'url_e', 'url_f', 'url_g']
    }
    sample_df = pd.DataFrame(data)

    try:
        if not os.path.exists('../data'):
            os.makedirs('../data')
        dummy_file = '../data/short_goodreads_data.csv'
        if not os.path.exists(dummy_file):
             sample_df.to_csv(dummy_file) 
             print(f"Created dummy file for testing: {dummy_file}")
    except Exception as e:
        print(f"Could not create dummy data file: {e}")


    print("\nInitializing Recommender with sample data...")
    try:
        recommender = ContentRecommender(sample_df)

        # Test getting recommendations
        book_title_to_recommend = '1984'
        print(f"\nGetting recommendations for: '{book_title_to_recommend}'")
        recommendations = recommender.get_recommendations(book_title_to_recommend, k=3, rating_weight=0.2)

        if not recommendations.empty:
            print("\nRecommendations:")
            print(recommendations)
        else:
            print("No recommendations found.")

        # Test for a non-existent book
        print(f"\nGetting recommendations for: 'Non Existent Book'")
        recommendations_non_existent = recommender.get_recommendations('Non Existent Book', k=3)
        if recommendations_non_existent.empty:
             print("Correctly handled non-existent book.")


    except Exception as e:
        print(f"An error occurred during recommender testing: {e}")
        import traceback
        traceback.print_exc()


    print("\nContent recommender module executed (for testing).")