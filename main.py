import pandas as pd
import os
import random

from data_loader import loader
from recommender.content_recommender import ContentRecommender
from evaluation import metrics
from visualization import plot_results

DATA_FILE_PATH = os.path.join('data', 'short_goodreads_data.csv')

# Recommendation parameters
TARGET_BOOK_TITLE = None # Set to a specific title, or None to pick randomly
NUM_RECOMMENDATIONS = 10
RATING_WEIGHT = 0.15 # How much weight to give ratings in the final score (0 to 1)

# Evaluation parameters (using proxy definition of relevance)
EVALUATION_K = 10 # Evaluate Precision/Recall/F1 @ k

# Visualization parameters
PLOT_SIMILARITY_HEATMAP = True
HEATMAP_SAMPLE_SIZE = 12
PLOT_RATING_DIST = True
PLOT_SIMILARITY_DIST = True

# --- Main Execution ---
def run_pipeline():
    """Executes the full recommendation system pipeline."""
    print("--- Starting Book Recommendation Pipeline ---")

    # 1. Load Data
    print("\n[Step 1/5] Loading Data...")
    try:
        df = loader.load_data(DATA_FILE_PATH)
        # Basic check after loading
        if df.empty:
            print("Error: Loaded DataFrame is empty. Exiting.")
            return
        if 'Book' not in df.columns:
             print("Error: 'Book' column not found in DataFrame. Check data loading. Exiting.")
             return
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}. Please check the path.")
        print("Exiting.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Exiting.")
        return

    # Ensure 'Book' column has unique enough identifiers or handle duplicates
    if df['Book'].duplicated().any():
        print("Warning: Duplicate book titles found. Recommendations might be inconsistent for duplicates.")
        # Optionally, handle duplicates here (e.g., add Author to title, or drop duplicates)

    # 2. Build Recommender (includes preprocessing and model building)
    print("\n[Step 2/5] Initializing Recommender and Building Model...")
    try:
        recommender = ContentRecommender(df)
        if recommender.cosine_sim is None:
             print("Error: Model building failed (cosine_sim is None). Check logs. Exiting.")
             return
    except Exception as e:
        print(f"Error initializing or building recommender: {e}")
        import traceback
        traceback.print_exc()
        print("Exiting.")
        return

    # 3. Get Recommendations
    print("\n[Step 3/5] Getting Recommendations...")

    # Select a book title
    if TARGET_BOOK_TITLE is None or TARGET_BOOK_TITLE not in recommender.indices:
        if TARGET_BOOK_TITLE is not None: # If specified but not found
             print(f"Warning: Target book '{TARGET_BOOK_TITLE}' not found. Selecting a random book.")
        # Select a random book title from the available indices
        available_titles = recommender.indices.index.tolist()
        if not available_titles:
            print("Error: No book titles available in the recommender indices. Exiting.")
            return
        target_book_title = random.choice(available_titles)
        print(f"No target book specified or found, randomly selected: '{target_book_title}'")
    else:
        target_book_title = TARGET_BOOK_TITLE
        print(f"Target book: '{target_book_title}'")

    # Get recommendations
    try:
        recommendations_df = recommender.get_recommendations(
            target_book_title,
            k=NUM_RECOMMENDATIONS,
            rating_weight=RATING_WEIGHT
        )
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        recommendations_df = pd.DataFrame() # Ensure it's an empty df on error

    if recommendations_df.empty:
        print(f"Could not generate recommendations for '{target_book_title}'.")
    else:
        print(f"\nTop {NUM_RECOMMENDATIONS} recommendations for '{target_book_title}':")
        # Display recommendations (adjust columns as needed)
        display_cols = ['Book', 'Author', 'Avg_Rating', 'similarity_score']
        if 'combined_score' in recommendations_df.columns:
             display_cols.append('combined_score')
        print(recommendations_df[display_cols].to_string(index=False))


    # 4. Evaluate Recommendations (Example Implementation)
    print("\n[Step 4/5] Evaluating Recommendations (Example)...")
    if not recommendations_df.empty:
        try:
            # --- Example: Define relevant items based on shared genres ---
            # This is a PROXY for actual relevance. Replace with a real ground truth if available.
            target_book_details = recommender.df.iloc[recommender.indices[target_book_title]]
            target_genres = set(target_book_details['Genres']) # Assumes 'Genres' is a list

            if not target_genres:
                print("Warning: Target book has no genres listed. Cannot use genre-based relevance proxy.")
                relevant_items_proxy = set()
            else:
                relevant_items_proxy = set()
                for idx, book_genres in recommender.df['Genres'].items():
                     if isinstance(book_genres, (list, set)) and target_genres.intersection(book_genres):
                         # Use the book title as the item identifier
                         book_title_for_eval = recommender.df.loc[idx, 'Book']
                         # Avoid adding the target book itself to the relevant set
                         if book_title_for_eval != target_book_title:
                             relevant_items_proxy.add(book_title_for_eval)


            recommended_titles = recommendations_df['Book'].tolist()

            print(f"Target Book Genres: {target_genres}")
            print(f"Proxy Relevant Items (Shared Genre, sample): {list(relevant_items_proxy)[:10]}...")
            print(f"Total Proxy Relevant Items Found: {len(relevant_items_proxy)}")


            precision = metrics.precision_at_k(recommended_titles, relevant_items_proxy, EVALUATION_K)
            recall = metrics.recall_at_k(recommended_titles, relevant_items_proxy, EVALUATION_K)
            f1 = metrics.f1_score_at_k(recommended_titles, relevant_items_proxy, EVALUATION_K)

            print(f"\nEvaluation Metrics @{EVALUATION_K} (using genre proxy):")
            print(f"  Precision@{EVALUATION_K}: {precision:.4f}")
            print(f"  Recall@{EVALUATION_K}:    {recall:.4f}")
            print(f"  F1-Score@{EVALUATION_K}:  {f1:.4f}")

        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Skipping evaluation as no recommendations were generated.")


    # 5. Visualize Results
    print("\n[Step 5/5] Visualizing Results...")
    try:
        if PLOT_SIMILARITY_HEATMAP:
            if recommender.cosine_sim is not None and recommender.cosine_sim.shape[0] > 1:
                 print("Plotting similarity heatmap (sample)...")
                 book_titles = recommender.df['Book'].tolist() # Get all book titles as labels
                 plot_results.plot_similarity_heatmap(
                     recommender.cosine_sim,
                     labels=book_titles,
                     sample_size=HEATMAP_SAMPLE_SIZE
                 )
            else:
                 print("Skipping similarity heatmap (matrix too small or not generated).")

        if not recommendations_df.empty:
            if PLOT_RATING_DIST:
                 print("Plotting rating distribution for recommendations...")
                 plot_results.plot_recommendation_ratings_distribution(recommendations_df)
            if PLOT_SIMILARITY_DIST:
                 print("Plotting similarity score distribution for recommendations...")
                 plot_results.plot_similarity_distribution(recommendations_df)
        else:
            print("Skipping recommendation-specific plots as no recommendations were generated.")

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Pipeline Execution Finished ---")


if __name__ == '__main__':
    run_pipeline()