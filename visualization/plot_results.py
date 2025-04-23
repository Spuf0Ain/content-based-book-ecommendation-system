# visualization/plot_results.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np # Needed for heatmap example

def plot_similarity_heatmap(similarity_matrix: np.ndarray, labels: list = None, sample_size: int = 10, title: str = "Book Similarity Heatmap"):
    """
    Plots a heatmap of the cosine similarity matrix for a sample of items.

    Args:
        similarity_matrix (np.ndarray): The square cosine similarity matrix.
        labels (list, optional): List of labels (e.g., book titles) corresponding
                                 to the rows/columns. Defaults to None.
        sample_size (int): The number of items to sample for the heatmap.
                           If the matrix size is smaller, uses the full size.
        title (str): The title for the plot.
    """
    if similarity_matrix is None or not isinstance(similarity_matrix, np.ndarray):
        print("Warning: Similarity matrix is None or invalid. Cannot plot heatmap.")
        return

    n_items = similarity_matrix.shape[0]
    if n_items == 0:
        print("Warning: Similarity matrix is empty. Cannot plot heatmap.")
        return

    sample_size = min(sample_size, n_items)

    sample_indices = np.random.choice(n_items, sample_size, replace=False)
    sample_matrix = similarity_matrix[np.ix_(sample_indices, sample_indices)]

    sample_labels = None
    if labels is not None and len(labels) == n_items:
        sample_labels = [labels[i] for i in sample_indices]

    plt.figure(figsize=(10, 8))
    sns.heatmap(sample_matrix, annot=True, cmap='viridis', fmt=".2f",
                xticklabels=sample_labels or sample_indices,
                yticklabels=sample_labels or sample_indices)
    plt.title(title + f" (Sample of {sample_size})")
    plt.xlabel("Books")
    plt.ylabel("Books")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_recommendation_ratings_distribution(recommendations_df: pd.DataFrame, rating_col: str = 'Avg_Rating', title: str = "Distribution of Ratings for Recommended Books"):
    """
    Plots the distribution of average ratings for the recommended books.

    Args:
        recommendations_df (pd.DataFrame): DataFrame containing recommended books.
                                          Must include the 'Avg_Rating' column.
        rating_col(str): Name of the column containing ratings.
        title (str): The title for the plot.
    """
    if recommendations_df is None or recommendations_df.empty:
        print("Warning: Recommendations DataFrame is empty. Cannot plot rating distribution.")
        return

    if rating_col not in recommendations_df.columns:
        print(f"Warning: Column '{rating_col}' not found in recommendations DataFrame. Cannot plot rating distribution.")
        return

    # Drop rows with missing ratings before plotting
    ratings = recommendations_df[rating_col].dropna()

    if ratings.empty:
        print(f"Warning: No valid ratings found in column '{rating_col}'. Cannot plot distribution.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(ratings, kde=True, bins=10) # Use histplot for distribution
    plt.title(title)
    plt.xlabel("Average Rating")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_similarity_distribution(recommendations_df: pd.DataFrame, similarity_col: str = 'similarity_score', title: str = "Distribution of Similarity Scores for Recommended Books"):
    """
    Plots the distribution of similarity scores for the recommended books.

    Args:
        recommendations_df (pd.DataFrame): DataFrame containing recommended books.
                                          Must include the similarity score column.
        similarity_col (str): Name of the column containing similarity scores.
        title (str): The title for the plot.
    """
    if recommendations_df is None or recommendations_df.empty:
        print("Warning: Recommendations DataFrame is empty. Cannot plot similarity distribution.")
        return

    if similarity_col not in recommendations_df.columns:
        print(f"Warning: Column '{similarity_col}' not found in recommendations DataFrame. Cannot plot similarity distribution.")
        return

    # Drop rows with missing similarity scores before plotting
    similarity_scores = recommendations_df[similarity_col].dropna()

    if similarity_scores.empty:
        print(f"Warning: No valid similarity scores found in column '{similarity_col}'. Cannot plot distribution.")
        return

    plt.figure(figsize=(8, 5))
    sns.histplot(similarity_scores, kde=True, bins=10)
    plt.title(title)
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    sample_sim_matrix = np.random.rand(15, 15)
    # Make it symmetric and set diagonal to 1
    sample_sim_matrix = (sample_sim_matrix + sample_sim_matrix.T) / 2
    np.fill_diagonal(sample_sim_matrix, 1)
    sample_labels = [f'Book {i+1}' for i in range(15)]

    print("Plotting similarity heatmap...")
    plot_similarity_heatmap(sample_sim_matrix, labels=sample_labels, sample_size=8)

    rec_data = {
        'Book': [f'Rec Book {i}' for i in range(10)],
        'Author': [f'Author {i}' for i in range(10)],
        'Avg_Rating': np.random.uniform(3.0, 5.0, 10).round(1),
        'Num_Ratings': np.random.randint(50, 1000, 10),
        'similarity_score': np.random.uniform(0.5, 0.9, 10).round(3),
        'Genres': [['Fiction']]*10,
        'URL': ['url']*10
    }
    sample_recs_df = pd.DataFrame(rec_data)
    sample_recs_df.loc[3, 'Avg_Rating'] = np.nan

    print("\nPlotting recommended books rating distribution...")
    plot_recommendation_ratings_distribution(sample_recs_df)

    print("\nPlotting recommended books similarity score distribution...")
    plot_similarity_distribution(sample_recs_df)

    print("\nTesting with empty DataFrame...")
    empty_df = pd.DataFrame()
    plot_recommendation_ratings_distribution(empty_df)
    plot_similarity_distribution(empty_df)

    print("\nPlot results module executed successfully (for testing).")